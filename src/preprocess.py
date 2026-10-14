"""전처리 — 사용자가 정한 결측 처리·열 삭제·이상치 제거만 적용 (10/12 결정, 결정기록.md 8~9번, 위경도 유지는 19번)

입력: data/processed/features_all.csv, split.csv, input_all.csv(지번: 주변 시세 1km 재계산용),
      data/raw/trades_rh.csv(거래 유형: 이상치 ④ 판단에만 사용, 변수로는 안 씀)
출력: data/processed/preprocessed_all.csv, fill_values.json (빈칸 채움에 쓴 동·구 중앙값 — predict.py가 같은 값으로 채움),
      retrain_exclude.json (시험용 중 학습용과 같은 이상치 규칙에 걸리는 id — 전체 재학습 때 predict.py가 뺌, 10/13 추가)
하지 않은 것: 인코딩(5단계에서). 타깃 변환 안 함·파생변수 안 만듦·구/동 둘 다 유지 (사용자 결정)
동 중앙값은 학습용(train) 행에서만 계산해 시험용 행에도 같은 값을 씀 → 시험 데이터 정보가 학습에 새지 않게
이상치 제거는 학습용 행에만 → 시험용은 회사 검증처럼 걸러지지 않은 실제 거래 그대로
실행: python src/preprocess.py
"""
import json
from pathlib import Path

import pandas as pd

from features import load_sources, nearby

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"


FILL = {}  # 채움 값 기록 → fill_values.json (predict.py가 학습 때와 같은 값으로 채우도록, 10/13 추가)
EXCLUDE = []  # 시험용 중 이상치 규칙 해당 id → retrain_exclude.json (9단계 평가 뒤 전체 재학습, 10/13 사용자 결정)


def dong_median_fill(df, col, is_train):
    """빈칸을 같은 동의 중앙값(학습용 행으로 계산)으로 채움. 구 중앙값도 기록 (predict.py에서 학습용에 없는 동일 때 사용)"""
    med = df[is_train].groupby("dong")[col].median()
    FILL[col] = {"dong": med.dropna().to_dict(), "gu": df[is_train].groupby("gu")[col].median().to_dict()}
    return df[col].fillna(df.dong.map(med))


def preprocess(df, is_train):
    df = df.copy()
    log = {}

    # ① 좌표가 없는 행 삭제 — lat·lon·역거리·노선 수·주변 거래 건수가 같은 행에서 비어 있음
    drop = df[["lat", "lon", "station_dist_m", "n_lines_500m", "nb_cnt_500m"]].isna().any(axis=1)
    log["① 행 삭제 (좌표 없음)"] = int(drop.sum())
    df, is_train = df[~drop].copy(), is_train[~drop]

    # ② 열 삭제
    df = df.drop(columns=["top_floor", "land_per_hh"])
    log["② 열 삭제"] = "top_floor, land_per_hh"

    # ③ 주변 시세: 반경 500m·12개월 안 다른 지번 거래 없음 → 1km로 다시 계산 → 그래도 없으면 동 중앙값
    miss = df.nb_ppm_500m.isna()
    sub = df[miss]
    ppm_1km, _ = nearby((sub.gu + "|" + sub.dong + "|" + sub.jibun).to_numpy(), sub.lat.to_numpy(), sub.lon.to_numpy(),
                        pd.to_datetime(sub.deal_date).to_numpy(), load_sources(), radius=1000)
    df.loc[miss, "nb_ppm_500m"] = ppm_1km
    still = df.nb_ppm_500m.isna()
    df["nb_ppm_500m"] = dong_median_fill(df, "nb_ppm_500m", is_train)
    log["③ 주변 시세 빈칸"] = f"{int(miss.sum())}건 → 1km로 {int(miss.sum() - still.sum())}건 채움, 동 중앙값 {int(still.sum())}건"

    # ④ 동 중앙값 + 빈칸 표시 칸 (1 = 원래 빈칸이었음)
    for col in ["pub_ppm", "hhld_cnt", "parking_per_hh"]:
        df[f"{col}_missing"] = df[col].isna().astype(int)
        df[col] = dong_median_fill(df, col, is_train)
        log[f"④ {col}"] = f"{int(df[f'{col}_missing'].sum())}건 동 중앙값 + {col}_missing 칸"

    # ⑤ 건물 나이: 동 중앙값 (표시 칸 없음)
    log["⑤ building_age"] = f"{int(df.building_age.isna().sum())}건 동 중앙값"
    df["building_age"] = dong_median_fill(df, "building_age", is_train)

    # ⑥ 승강기 0 (다세대 대부분 승강기 없음), 구조 '모름'
    log["⑥ elevator"] = f"{int(df.elevator.isna().sum())}건 0으로"
    df["elevator"] = df.elevator.fillna(0)
    log["⑥ structure"] = f"{int(df.structure.isna().sum())}건 '모름'으로"
    df["structure"] = df.structure.fillna("모름")

    # ⑦ 이상치 제거 — 학습용 행에만. 결측 처리 뒤에 해서 위에서 채운 동 중앙값은 바뀌지 않음
    top = is_train & (df.price_won == df[is_train].price_won.max()).to_numpy()  # 최고가 1건 (대치동 지하 55억)
    q01 = df[is_train].groupby("gu").price_won.quantile(.01)  # 권역별 거래가 하위 1% 경계 (학습용으로 계산)
    low = is_train & ((df.price_won <= df.gu.map(q01)) & (df.dealingGbn == "직거래")).to_numpy()
    log["⑦ 이상치 (학습용만)"] = f"최고가 {int(top.sum())}건, 권역 하위 1% 중 직거래 {int(low.sum())}건 삭제"
    # 시험용 행은 지우지 않고 id만 기록 (같은 경계·같은 규칙). 최고가 규칙은 시험용 최고가가 학습용 최고가보다 낮아 해당 없음
    test_low = ~is_train & ((df.price_won <= df.gu.map(q01)) & (df.dealingGbn == "직거래")).to_numpy()
    EXCLUDE.extend(df.id[test_low].tolist())
    log["⑦ 시험용 중 같은 규칙 해당 (재학습 때 제외)"] = f"{len(EXCLUDE)}건"
    df = df[~(top | low)]

    # ⑧ 위경도는 남김 (처음엔 '구로 97%·91% 설명됨'으로 삭제 → 제출 뒤 실험에서 넣으면 교차검증 14.40 → 13.89%, 결정기록.md 19번)
    # 같은 건물 과거 시세(sb_sim)는 빈칸 그대로 = 과거 거래 없음. 트리 모델만 쓰는 칸이라 채우지 않음
    log["⑧ 같은 건물 과거 시세 빈칸 (채우지 않음)"] = f"{int(df.sb_sim.isna().sum())}건"
    return df, log


if __name__ == "__main__":
    feat = pd.read_csv(PROC / "features_all.csv", float_precision="round_trip")  # 소수 끝자리까지 원본 그대로 읽기
    split = pd.read_csv(PROC / "split.csv")
    jibun = pd.read_csv(PROC / "input_all.csv", dtype=str)[["id", "jibun"]].astype({"id": int})
    t = pd.read_csv(PROC.parent / "raw" / "trades_rh.csv", dtype=str)
    t = t[((t.gu != "강서구") | (t.umdNm == "화곡동")) & (t.houseType == "다세대")
          & (t.cdealType.fillna("").str.strip() != "O")].reset_index(drop=True)
    t["id"] = range(1, len(t) + 1)  # make_input.py와 같은 필터·순서 → 같은 id
    df = feat.merge(split, on="id").merge(jibun, on="id").merge(t[["id", "dealingGbn"]], on="id")
    out, log = preprocess(df, (df.set == "train").to_numpy())
    out = out.drop(columns=["set", "jibun", "dealingGbn"])  # 계산에만 쓴 칸 (분할은 split.csv, 지번·거래 유형은 원본에 있음)
    out.to_csv(PROC / "preprocessed_all.csv", index=False, encoding="utf-8-sig")
    (PROC / "fill_values.json").write_text(json.dumps(FILL, ensure_ascii=False, indent=1), encoding="utf-8")
    (PROC / "retrain_exclude.json").write_text(json.dumps({"id": EXCLUDE}), encoding="utf-8")
    for k, v in log.items():
        print(k, ":", v)
    print(f"저장: preprocessed_all.csv ({len(out):,}행 × {out.shape[1]}열)")
