"""후보 변수(피처) 만들기 — 학습과 predict.py가 같은 함수를 쓴다

입력: 과제 input.csv 형식 행(id, sigungu, dong, jibun, floor, ho, area_m2) + 행마다 기준일
  - 학습: 기준일 = 실제 거래일 (그 거래 시점에 알 수 있던 정보만 쓰기 위해)
  - 예측: 기준일 = 산출 기준일
  - 공시가격도 기준일에 이미 공개돼 있던 연도 값만 씀 (2026 값을 2025년 거래에 붙이면 미래 정보 = 시점 누수)
후보 선정 근거(감정평가 거래사례비교법의 요인)는 결정기록.md 4번
실행: python src/features.py → data/processed/features_all.csv (EDA용, 정답 가격 포함)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
KEY = ["gu", "umdNm", "jibun"]
START = pd.Timestamp("2023-10-01")  # 수집 시작월. 시점 변수 = 여기서 몇 개월 지났는지
RADIUS = 500  # 주변 시세·역세권 반경(m). 과제 basis 예시("반경 500m")를 따름, 튜닝 단계에서 조정 가능
PUB_YEARS = [2023, 2024, 2025, 2026]  # 받아 둔 공시가격 연도 (data/raw/house_price_{연도}.csv)


def pub_year(ref):
    """기준일에 이미 공개돼 있던 가장 최근 공시가격 연도. 정기 공시일은 매년 4월 30일경 (결정기록.md 7번)
    → 4월 30일 이후면 그해 값, 그 전이면 전년도 값. 거래 뒤에 공시된 값을 쓰면 시점 누수"""
    after = (ref.dt.month > 4) | ((ref.dt.month == 4) & (ref.dt.day >= 30))
    return np.where(after, ref.dt.year, ref.dt.year - 1).clip(max=max(PUB_YEARS))


def dist_m(lat1, lon1, lat2, lon2):
    """두 좌표 사이 직선거리(m), 하버사인 공식 (지구를 공으로 보고 계산)"""
    p = np.pi / 180
    h = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12_742_000 * np.arcsin(np.sqrt(h))


def building_table(t):
    """건축물대장 표제부 원본 행 → 지번별 1행. predict.py가 실행 중 조회한 지번에도 같은 규칙을 쓰도록 함수로 분리 (10/13)
    주건축물만. 한 지번에 동이 여러 개면 공동주택 동만 남기고 합침 (307개 지번)"""
    t = t[t["_error"].isna() & (t.mainAtchGbCdNm == "주건축물")]
    has_apt = t.groupby(KEY).mainPurpsCdNm.transform(lambda s: (s == "공동주택").any())
    t = t[(t.mainPurpsCdNm == "공동주택") | ~has_apt].copy()
    num = lambda c: pd.to_numeric(t[c], errors="coerce")
    t["apr"] = pd.to_datetime(t.useAprDay, format="%Y%m%d", errors="coerce")
    t["park"] = num("indrAutoUtcnt") + num("oudrAutoUtcnt") + num("indrMechUtcnt") + num("oudrMechUtcnt")
    t["grnd"], t["hhld"], t["elev"], t["plat"] = num("grndFlrCnt"), num("hhldCnt"), num("rideUseElvtCnt"), num("platArea")
    return t.groupby(KEY).agg(apr=("apr", "max"), grnd=("grnd", "max"), hhld=("hhld", "sum"), elev=("elev", "sum"),
                              park=("park", "sum"), plat=("plat", "max"), structure=("strctCdNm", "first"))


def pub_table(p):
    """공시가격 원본 행 → {지번: 그 지번의 호별 공시가격}. predict.py도 같은 규칙을 쓰도록 함수로 분리 (10/13)
    API가 같은 호를 2번씩 줌. 2024·2025년은 갱신일(lastUpdtDt)만 다른 2행이라 그 칸을 빼고 중복 제거"""
    p = p[p["_error"].isna()].drop(columns="lastUpdtDt").drop_duplicates()
    p["ar"] = p.prvuseAr.astype(float).round(2)
    p["ppm"] = p.pblntfPc.astype(float) / p.prvuseAr.astype(float)
    return {k: grp for k, grp in p.groupby(KEY)}


def load_sources():
    src = {}
    g = pd.read_csv(RAW / "geocode_jibun.csv", dtype=str)
    src["geo"] = g[g.status == "OK"].set_index(KEY)[["lat", "lon"]].astype(float)
    src["bld"] = building_table(pd.read_csv(RAW / "building_title.csv", dtype=str))

    tr = pd.read_csv(RAW / "trades_rh.csv", dtype=str)  # 대장에 사용승인일이 없을 때 쓰는 건축년도
    src["build_year"] = pd.to_numeric(tr.buildYear, errors="coerce").groupby([tr.gu, tr.umdNm, tr.jibun]).max()

    src["pub"] = {year: pub_table(pd.read_csv(RAW / f"house_price_{year}.csv", dtype=str)) for year in PUB_YEARS}  # 연도 → {지번: 표}

    s = pd.read_csv(RAW / "subway_stations.csv")
    src["st"] = s[["LAT", "LOT"]].to_numpy()
    src["st_route"] = pd.get_dummies(s.ROUTE).to_numpy(int)  # 역 × 노선 표 (환승역은 노선마다 1행)

    # 주변 시세 재료: 다세대 실거래 전체 (좌표·거래일·㎡당 가격)
    pool = pd.read_csv(PROC / "input_all.csv", dtype=str).merge(pd.read_csv(PROC / "answer_all.csv", dtype=str), on="id")
    pool["gu"], pool["umdNm"] = pool.sigungu.str.split().str[1], pool.dong
    pool = pool.join(src["geo"], on=KEY).dropna(subset=["lat"])
    pool["date"] = pd.to_datetime(pool.deal_date)
    pool["ppm"] = pool.price_won.astype(float) / pool.area_m2.astype(float)
    src["pool"] = pool
    return src


def pub_lookup(grp, floor, ho, area):
    """공시가격 ㎡당 값: 같은 지번에서 층 → 호 → 면적 순으로 좁혀 중앙값. 좁힌 결과가 비면 그 단계는 건너뜀"""
    if grp is None:
        return np.nan, "없음"
    level = "지번"
    for col, val, name in [("floorNm", floor, "층"), ("hoNm", ho, "호"), ("ar", area, "면적")]:
        if val is None or val == "" or (isinstance(val, float) and np.isnan(val)):
            continue
        narrowed = grp[grp[col] == val]
        if len(narrowed):
            grp, level = narrowed, level + "+" + name
    return grp.ppm.median(), level


def make_features(rows, ref_dates, src=None):
    src = src or load_sources()
    df = rows.reset_index(drop=True).copy()
    df["gu"], df["umdNm"] = df.sigungu.str.split().str[1], df.dong
    ref = pd.to_datetime(pd.Series(ref_dates)).reset_index(drop=True)
    idx = pd.MultiIndex.from_frame(df[KEY])
    floor = pd.to_numeric(df.floor, errors="coerce")
    area = pd.to_numeric(df.area_m2, errors="coerce")
    b = src["bld"].reindex(idx).reset_index(drop=True)
    geo = src["geo"].reindex(idx).reset_index(drop=True)

    out = pd.DataFrame({"id": df.id, "gu": df.gu, "dong": df.dong, "area_m2": area, "floor": floor})
    out["top_floor"] = np.where(b.grnd.isna(), np.nan, (floor == b.grnd).astype(float))  # 꼭대기층 여부
    age = (ref - b.apr).dt.days / 365.25
    age = age.fillna(ref.dt.year - src["build_year"].reindex(idx).reset_index(drop=True))  # 대장 없으면 실거래 건축년도
    out["building_age"] = age.where(age >= 0)  # 음수 = 거래 뒤에 새로 지은 건물 → 거래 당시 정보 아님
    out["lat"], out["lon"] = geo.lat, geo.lon

    d_st = dist_m(geo.lat.to_numpy()[:, None], geo.lon.to_numpy()[:, None], src["st"][:, 0], src["st"][:, 1])
    out["station_dist_m"] = np.nanmin(np.where(np.isnan(d_st), np.inf, d_st), axis=1)
    out.loc[geo.lat.isna(), "station_dist_m"] = np.nan
    out["n_lines_500m"] = (((d_st <= RADIUS).astype(int) @ src["st_route"]) > 0).sum(axis=1)  # 반경 안 노선 수
    out.loc[geo.lat.isna(), "n_lines_500m"] = np.nan

    # 면적 반올림은 공시가격 쪽과 같은 pandas .round(2)로 (파이썬 round()와 드물게 결과가 달라 매칭이 빠짐)
    # 공시가격 연도는 기준일 시점의 최신 공시 연도. 그해 목록에 없으면(예: 신축) 빈칸 — 더 뒤 연도 값으로 채우지 않음
    pub = [pub_lookup(src["pub"].get(y, {}).get(k), str(int(f)) if pd.notna(f) else None, h, a if pd.notna(a) else None)
           for k, y, f, h, a in zip(idx, pub_year(ref), floor, df.ho.fillna(""), area.round(2))]
    out["pub_ppm"], out["pub_level"] = [v for v, _ in pub], [lv for _, lv in pub]

    out["hhld_cnt"] = b.hhld.where(b.hhld > 0)
    out["elevator"] = b.elev
    out["parking_per_hh"] = b.park / out.hhld_cnt
    out["structure"] = b.structure
    out["land_per_hh"] = b.plat.where(b.plat > 0) / out.hhld_cnt  # 세대당 대지면적(대지지분 근사)
    out["t_month"] = (ref.dt.year - START.year) * 12 + ref.dt.month - START.month

    my_key = (df.gu + "|" + df.umdNm + "|" + df.jibun).to_numpy()
    out["nb_ppm_500m"], out["nb_cnt_500m"] = nearby(my_key, geo.lat.to_numpy(), geo.lon.to_numpy(), ref.to_numpy(), src)
    return out


def nearby(my_key, lat, lon, ref, src, radius=RADIUS):
    """주변 시세: 반경 radius(m) · 기준일 직전 12개월 · 다른 지번 거래의 ㎡당 가격 중앙값과 거래 건수
    같은 건물 거래는 정답이 섞일 수 있어 제외. 좌표가 없으면 둘 다 빈칸.
    반경을 인자로 둔 이유: 500m 안에 거래가 없을 때 전처리에서 1km로 다시 계산 (결정기록.md 8번)"""
    pool = src["pool"]
    pl_lat, pl_lon, pl_date, pl_ppm = pool.lat.to_numpy(), pool.lon.to_numpy(), pool.date.to_numpy(), pool.ppm.to_numpy()
    pl_key = (pool.gu + "|" + pool.umdNm + "|" + pool.jibun).to_numpy()
    nb_ppm, nb_cnt = np.full(len(lat), np.nan), np.zeros(len(lat))
    for i in range(len(lat)):
        if np.isnan(lat[i]):
            nb_cnt[i] = np.nan
            continue
        r = pd.Timestamp(ref[i])
        m = ((pl_date < r.to_datetime64()) & (pl_date >= (r - pd.DateOffset(months=12)).to_datetime64()) & (pl_key != my_key[i])
             & (dist_m(lat[i], lon[i], pl_lat, pl_lon) <= radius))
        nb_cnt[i] = m.sum()
        if m.any():
            nb_ppm[i] = np.median(pl_ppm[m])
    return nb_ppm, nb_cnt


if __name__ == "__main__":
    inp = pd.read_csv(PROC / "input_all.csv", dtype=str, keep_default_na=False)
    ans = pd.read_csv(PROC / "answer_all.csv", dtype=str)
    feat = make_features(inp, ans.deal_date)
    feat = feat.merge(ans, on="id")
    feat["price_won"] = feat.price_won.astype(float)
    feat.to_csv(PROC / "features_all.csv", index=False, encoding="utf-8-sig")
    print(f"저장: features_all.csv ({len(feat):,}행 × {feat.shape[1]}열)")
