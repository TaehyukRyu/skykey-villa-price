"""다세대(빌라) 시세 산출 — 과제 03번 입출력 규격 (결정기록.md 16번)

실행: python predict.py --input input.csv --output output.csv [--base-date 2026-10-06] [--train-only]
  입력에 base_date 칸(선택)이 있으면 행마다 그 날짜를 기준일로 씀 — 9단계 시험용 평가(기준일 = 각 거래일, 사용자 결정)용. 과제 규격 입력엔 없음
  --train-only: 학습용 8,793건만으로 학습 — 9단계 평가 재현용 (기본은 시험용까지 합친 10,974건)
입력: id, sigungu, dong, jibun, floor, ho(없을 수 있음), area_m2(비어 있을 수 있음)
출력: id, price_est, price_low, price_high, confidence, basis, status

흐름
 1. 입력 검사: 권역(강남구·관악구 전체, 강서구는 화곡동만), 지번 형식, 층 → 못 쓰면 status=fail, 사유는 basis (사용자 결정)
 2. 미리 수집하지 않은 지번은 실행 중 조회: 좌표(VWorld) → 건축물대장 표제부(건축HUB) → 공시가격(VWorld)
    API 키(DATA_GO_KR_KEY, VWORLD_KEY)는 환경변수 또는 .env. 키가 없으면 그 물건만 fail
 3. 호 표기 정리: 공시가격 목록의 호와 그대로 비교 → 안 맞으면 '호' 떼고 지하·지층·비 → B로 바꿔 다시 비교
 4. 면적 빈칸 (사용자 결정: 공시가격 목록): 같은 층·호 → 같은 층 중앙값 → 같은 지번 중앙값 → 같은 동 학습 거래 중앙값. 출처는 basis에 적음
 5. 피처: src/features.py make_features (학습과 같은 함수), 기준일 = 산출 기준일 (사용자 결정: 과제 안내일 2026-10-06)
 6. 빈칸 채움: 전처리와 같은 값 (data/processed/fill_values.json — 동 중앙값, 학습용에 없는 동이면 구 중앙값)
 7. 모델: 실행할 때마다 CatBoost·XGBoost·LightGBM 학습 (사용자 결정, 튜닝 설정·난수 고정 → 매번 같은 모델)
    학습 데이터 = 학습용 8,793 + 시험용 2,181 = 10,974건 (9단계 평가 뒤 전체 재학습, 사용자 결정. 시험용 중 학습용과 같은
    이상치 규칙에 걸린 19건은 data/processed/retrain_exclude.json으로 뺌)
    추정 = CatBoost (결정기록.md 13번), 나머지 둘은 confidence의 '세 모델 엇갈림'에만 씀
 8. confidence·구간: reports/uncertainty_params.json (결정기록.md 14번), basis: src/basis.py 템플릿 (15번, 기준가는 비슷한 면적 거래)
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

import collect_buildings  # noqa: E402  (src 경로를 넣은 뒤 import)
import collect_geocode  # noqa: E402
from basis import similar_area, template  # noqa: E402
from features import KEY, building_table, load_sources, make_features, nearby, pub_table, pub_year  # noqa: E402
from tune_models import build  # noqa: E402  (튜닝 때와 같은 모델 구성)

PROC, REPORT = ROOT / "data" / "processed", ROOT / "reports"
BASE_DATE = "2026-10-06"  # 산출 기준일 = 과제 안내일 (10/13 사용자 확인)
REGION = {"강남구": None, "관악구": None, "강서구": {"화곡동"}}  # None = 구 전체
TREES = ["CatBoost", "XGBoost", "LightGBM"]
OUT_COLS = ["id", "price_est", "price_low", "price_high", "confidence", "basis", "status"]


def read_input(path):
    """한글 CSV는 UTF-8(BOM 포함) 또는 엑셀 기본 CP949일 수 있어 둘 다 시도"""
    for enc in ("utf-8-sig", "cp949"):
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    df.columns = df.columns.str.strip()
    for c in ["ho", "area_m2"]:
        if c not in df:
            df[c] = ""
    return df.apply(lambda s: s.str.strip())


def parse_floor(s):
    """'3' → 3, 'B1'·'지하1'·'지1' → -1 (학습 데이터·공시가격 목록 모두 지하를 음수로 적음). 못 읽으면 None"""
    s = s.replace("층", "").replace(" ", "")
    m = re.fullmatch(r"(B|지하|지)(\d+)", s, flags=re.I)
    if m:
        return -int(m.group(2))
    return int(s) if re.fullmatch(r"-?\d+", s) else None


def check(row):
    """입력 검사 → (gu, floor, 실패 사유 또는 None)"""
    gu = next((t for t in row.sigungu.split() if t.endswith("구")), "")
    if gu not in REGION or (REGION[gu] is not None and row.dong not in REGION[gu]):
        return gu, None, f"권역 밖 주소({row.sigungu} {row.dong}) — 대상은 강남구·관악구·강서구 화곡동"
    if not re.fullmatch(r"\d+(-\d+)?", row.jibun):
        return gu, None, f"지번 형식을 읽지 못함({row.jibun}) — 예: 123-45 (산 지번은 미지원)"
    floor = parse_floor(row.floor)
    if floor is None:
        return gu, None, f"층을 읽지 못함({row.floor or '빈칸'})"
    return gu, floor, None


def lookup_new(gu, dong, jibun, src, year):
    """수집 안 된 지번을 실행 중 조회해 src에 추가 → 실패 사유 또는 None"""
    if not (os.environ.get("VWORLD_KEY") and os.environ.get("DATA_GO_KR_KEY")):
        return "미리 수집하지 않은 지번이라 실행 중 조회가 필요한데 API 키(VWORLD_KEY, DATA_GO_KR_KEY)가 없음"
    g = collect_geocode.geocode(gu, dong, jibun)
    if g["status"] != "OK" or not g["pnu"]:
        return f"지번 좌표를 찾지 못함 (VWorld: {g['status']} {g['error']})".strip()
    key = (gu, dong, jibun)
    src["geo"].loc[key, ["lat", "lon"]] = [float(g["lat"]), float(g["lon"])]
    bun, ji = (jibun.split("-") + ["0"])[:2]
    t = dict(gu=gu, umdNm=dong, jibun=jibun, dong_code=g["pnu"][:10], bun=bun.zfill(4), ji=ji.zfill(4))
    t["pnu_jibun"] = t["dong_code"] + "1" + t["bun"] + t["ji"]  # 원래 지번으로 조회 (지오코더가 근처 필지로 대체해도)
    title = pd.DataFrame(collect_buildings.fetch_title(t))
    if "_error" not in title:  # 오류가 한 건도 없으면 이 칸이 안 생김 (정리 함수가 이 칸을 씀)
        title["_error"] = np.nan
    if title["_error"].isna().any():  # 대장 기록이 있을 때만 (없으면 빈칸 → 동 중앙값으로 채움)
        src["bld"] = pd.concat([src["bld"], building_table(title)])
    price = pd.DataFrame(collect_buildings.fetch_price(t, str(year)))
    for c in ["_error", "lastUpdtDt"]:
        if c not in price:
            price[c] = np.nan
    if price["_error"].isna().any():
        src["pub"].setdefault(year, {}).update(pub_table(price))
    return None


def norm_ho(s):
    """호 표기 정리: '301호' → '301', '지하B01'·'지층B01'·'비01'·'B1' → 'B1', '2층201호' → '201'"""
    s = re.sub(r"\s", "", str(s)).replace("호", "")
    s = s.split("층")[-1]
    under = bool(re.match(r"(지하|지층|비|지|B)", s, flags=re.I))
    digits = re.sub(r"\D", "", s)
    return ("B" if under else "") + str(int(digits)) if digits else s


def match_ho(ho, floor, grp):
    """입력 호 → 공시가격 목록에 실제로 적힌 호 문자열. 그대로 있으면 그대로, 정리해서 하나로 맞으면 그 표기, 아니면 입력 그대로"""
    if not ho or grp is None or (grp.hoNm == ho).any():
        return ho
    cand = grp[grp.floorNm == str(floor)] if (grp.floorNm == str(floor)).any() else grp
    hit = cand.hoNm[cand.hoNm.fillna("").map(norm_ho) == norm_ho(ho)].unique()
    return hit[0] if len(hit) == 1 else ho


def fill_area(area, ho, floor, grp, dong_area):
    """면적 빈칸 → (면적, basis에 적을 출처). 공시가격 목록(사용자 결정) → 동 학습 거래 중앙값"""
    if area:
        return float(area), ""
    if grp is not None and grp.ar.notna().any():
        fl = grp[grp.floorNm == str(floor)]
        same = fl[fl.hoNm == ho] if ho else fl.iloc[:0]
        if len(same):
            return same.ar.median(), "전용면적 빈칸 → 공시가격 목록의 같은 층·호 면적 사용"
        if len(fl):
            return fl.ar.median(), "전용면적 빈칸 → 공시가격 목록의 같은 층 면적 중앙값으로 추정"
        return grp.ar.median(), "전용면적 빈칸 → 공시가격 목록의 같은 지번 면적 중앙값으로 추정"
    return dong_area, "전용면적을 찾지 못해 같은 동 학습 거래의 면적 중앙값으로 추정"


def fill_missing(feat, src, refs):
    """전처리(src/preprocess.py)와 같은 규칙·같은 값으로 빈칸 채움. refs = 행마다 기준일"""
    F = json.loads((PROC / "fill_values.json").read_text(encoding="utf-8"))
    med = lambda col, dong, gu: F[col]["dong"].get(dong, F[col]["gu"][gu])
    miss = feat.nb_ppm_500m.isna()  # 반경 500m 거래 없음 → 1km → 그래도 없으면 동 중앙값
    if miss.any():
        sub = feat[miss]
        feat.loc[miss, "nb_ppm_500m"] = nearby((sub.gu + "|" + sub.dong + "|" + sub.jibun).to_numpy(), sub.lat.to_numpy(),
                                               sub.lon.to_numpy(), pd.to_datetime(refs[miss.to_numpy()]).to_numpy(), src, radius=1000)[0]
    for col in ["nb_ppm_500m", "pub_ppm", "hhld_cnt", "parking_per_hh", "building_age"]:
        if col in ["pub_ppm", "hhld_cnt", "parking_per_hh"]:
            feat[f"{col}_missing"] = feat[col].isna().astype(int)
        feat[col] = [v if pd.notna(v) else med(col, d, g) for v, d, g in zip(feat[col], feat.dong, feat.gu)]
    feat["elevator"] = feat.elevator.fillna(0)
    feat["structure"] = feat.structure.fillna("모름")
    return feat


def main():
    ap = argparse.ArgumentParser(description="다세대 시세 산출")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--base-date", default=BASE_DATE, help=f"산출 기준일 (기본 {BASE_DATE} = 과제 안내일)")
    ap.add_argument("--train-only", action="store_true", help="학습용 8,793건만으로 학습 (9단계 시험용 평가 재현용)")
    a = ap.parse_args()
    t0, inp = time.time(), read_input(a.input)
    per_row = "base_date" in inp and (inp.base_date != "").any()  # 행마다 기준일 (9단계 평가용)
    refs = pd.to_datetime(inp.base_date.where(inp.base_date != "", a.base_date) if per_row else pd.Series([a.base_date] * len(inp)))
    years = pub_year(refs)  # 기준일에 이미 공개된 공시가격 연도
    when = f"행마다(base_date 칸) {refs.min().date()}~{refs.max().date()}" if per_row else str(refs.iloc[0].date())
    print(f"입력 {len(inp)}건 · 기준일 {when} · 공시가격 {sorted(set(years.tolist()))}년", flush=True)

    src = load_sources()
    out = pd.DataFrame({"id": inp.id, "status": "ok", "basis": ""})
    rows, notes = [], {}
    for i, r in inp.iterrows():
        gu, floor, why = check(r)
        key, year = (gu, r.dong, r.jibun), int(years[i])
        if why is None and key not in src["geo"].index:
            print(f"  id {r.id}: 수집 안 된 지번 → 실행 중 조회 ({gu} {r.dong} {r.jibun})", flush=True)
            try:
                why = lookup_new(gu, r.dong, r.jibun, src, year)
            except Exception as e:  # 조회 중 예상 못 한 오류도 그 물건만 fail
                why = f"실행 중 조회 오류 ({type(e).__name__})"
        if why:
            out.loc[i, ["status", "basis"]] = ["fail", f"산출 불가: {why}"]
            continue
        grp = src["pub"].get(year, {}).get(key)
        ho = match_ho(r.ho, floor, grp)
        rows.append(dict(i=i, id=r.id, sigungu=f"서울특별시 {gu}", dong=r.dong, jibun=r.jibun, floor=str(floor), ho=ho, area_m2=r.area_m2, ref=refs[i], year=year))
    print(f"입력 검사·조회 끝: 산출 대상 {len(rows)}건, fail {int((out.status == 'fail').sum())}건 ({time.time() - t0:.0f}초)", flush=True)

    if rows:
        x = pd.DataFrame(rows)
        tr = pd.read_csv(PROC / "preprocessed_all.csv").merge(pd.read_csv(PROC / "split.csv"), on="id")
        if a.train_only:  # 9단계 평가 재현: 시험용이 학습에 섞이면 평가가 무의미해짐
            tr = tr[tr.set == "train"]
        else:  # 전체 재학습: 시험용 행은 전처리에서 이상치를 안 뺐으므로 같은 규칙에 걸린 19건을 여기서 뺌
            tr = tr[~tr.id.isin(json.loads((PROC / "retrain_exclude.json").read_text(encoding="utf-8"))["id"])]
        print(f"학습 데이터 {len(tr):,}건 ({'학습용만' if a.train_only else '학습용+시험용'})", flush=True)
        dong_area = tr.groupby("dong").area_m2.median()
        filled = [fill_area(r.area_m2, r.ho, r.floor, src["pub"].get(r.year, {}).get((r.sigungu.split()[1], r.dong, r.jibun)),
                            dong_area.get(r.dong, tr.area_m2.median())) for r in x.itertuples()]
        # 입력 면적은 문자열 그대로 (f"{:.2f}"로 다시 쓰면 29.935 → 29.93이 되어 학습 때 pandas .round(2)의 29.94와 달라짐, 9단계에서 발견)
        x["area_m2"] = [r.area_m2 if r.area_m2 else f"{a:.2f}" for r, (a, _) in zip(x.itertuples(), filled)]
        notes = dict(zip(x.i, [n for _, n in filled]))
        feat = make_features(x[["id", "sigungu", "dong", "jibun", "floor", "ho", "area_m2"]], list(x.ref), src)
        feat["jibun"] = x.jibun.to_numpy()
        no_geo = feat.lat.isna().to_numpy()  # 좌표가 없으면 역거리·주변 시세를 못 구함 (조회가 성공했으면 생기지 않음)
        sim_ppm, sim_cnt = similar_area((feat.gu + "|" + feat.dong + "|" + feat.jibun).to_numpy(), feat.lat.to_numpy(), feat.lon.to_numpy(),
                                        x.ref.to_numpy(), feat.area_m2.to_numpy(), src)  # basis 기준가용 (모델 입력 아님)
        feat = fill_missing(feat, src, x.ref.to_numpy())

        params = json.loads((REPORT / "best_params.json").read_text(encoding="utf-8"))
        logp = {}
        for n in TREES:  # 위 학습 데이터로 학습 (튜닝 설정, 난수 고정)
            logp[n] = np.log(build(n, params[n]).fit(tr, tr.price_won).predict(feat))
        print(f"모델 3개 학습·예측 끝 ({time.time() - t0:.0f}초)", flush=True)

        P = json.loads((REPORT / "uncertainty_params.json").read_text(encoding="utf-8"))
        est = np.exp(logp["CatBoost"])
        L = np.column_stack([logp[n] for n in TREES])
        spread = (np.exp(L.max(1) - L.min(1)) - 1) * 100
        X = np.column_stack([np.ones(len(feat)), np.log1p(feat.nb_cnt_500m), spread, feat.gu == "강남구", feat.gu == "관악구"]).astype(float)
        conf = 1 / (1 + np.exp(-X @ np.array(P["confidence"]["계수"])))
        tier = np.digitize(conf, P["등급"]["경계"])
        low, high = est * np.exp(np.array(P["구간"]["log_하한"])[tier]), est * np.exp(np.array(P["구간"]["log_상한"])[tier])

        for j, i in enumerate(x.i):
            if no_geo[j]:
                out.loc[i, ["status", "basis"]] = ["fail", "산출 불가: 좌표가 없어 주변 시세·역 거리를 계산하지 못함"]
                continue
            f = feat.iloc[j].to_dict() | {"est": est[j], "tier": P["등급"]["이름"][tier[j]], "spread_pct": spread[j], "area_note": notes[i],
                                          "sim_ppm": sim_ppm[j], "sim_cnt": sim_cnt[j]}
            out.loc[i, ["price_est", "price_low", "price_high"]] = [int(round(v, -4)) for v in (est[j], low[j], high[j])]
            out.loc[i, ["confidence", "basis"]] = [round(float(conf[j]), 2), template(f)]

    out = out.reindex(columns=OUT_COLS)
    for c in ["price_est", "price_low", "price_high"]:
        out[c] = out[c].astype("Int64")
    out.to_csv(a.output, index=False, encoding="utf-8-sig")
    print(f"저장: {a.output} · ok {int((out.status == 'ok').sum())}건 · fail {int((out.status == 'fail').sum())}건 · {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
