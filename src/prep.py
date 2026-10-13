"""모델별 입력 처리 순서 (A 방식: 원본 파일 1개 + 모델마다 처리 순서를 묶어 둠, 결정기록.md 10번)

preprocessed_all.csv(원래 단위)를 받아, 모델에 맞게 변환·인코딩한다 (10/12 사용자 결정).
처리 순서는 학습용으로만 fit(범주 목록 등 계산)하고, 시험용·predict.py에는 transform만 → 시험 정보가 새지 않음.
R의 recipes(step_log → step_dummy)와 같은 역할.

- regression_prep(): 선형회귀용. log 4개 적용, 나머지 수치형 그대로, 동·구조 원핫(기준 범주 1개 뺌), 구 제외
- tree_prep(): 트리 계열용. 수치형 그대로, 동·구조 라벨 인코딩, 구 제외
- log_target(): 타깃(거래가)을 log로 학습하고 예측은 exp로 원 단위로 되돌림 (회귀·트리 공통)
"""
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

NUM = ["area_m2", "floor", "building_age", "station_dist_m", "n_lines_500m", "pub_ppm", "hhld_cnt", "elevator",
       "parking_per_hh", "t_month", "nb_ppm_500m"]
# log 근거: 변환 전 왜도 1 이상인 연속값 → 변환 후 왜도가 0 쪽으로 줄어드는 것 확인 (결정기록.md 10번)
LOG_COLS = ["pub_ppm", "hhld_cnt", "parking_per_hh", "nb_ppm_500m"]
FLAGS = ["pub_ppm_missing", "hhld_cnt_missing", "parking_per_hh_missing"]
CAT = ["dong", "structure"]  # 구는 동과 완전히 겹쳐 모델 입력에서 뺌 (파일에는 남김)
BASE = [["화곡동"], ["철근콘크리트구조"]]  # 원핫 기준 범주 = 학습용에서 건수가 가장 많은 범주. 예측엔 영향 없고 계수 해석 기준
STRUCT_GROUP = {"철골콘크리트구조": "철골", "경량철골구조": "철골"}  # 소수 범주(8건·1건) 묶기
TARGET = "price_won"


def group_structure(X):
    return X.replace({"structure": STRUCT_GROUP})


def _cat(encoder):
    return Pipeline([("group", FunctionTransformer(group_structure, feature_names_out="one-to-one")), ("encode", encoder)])


def regression_prep(log_cols=LOG_COLS, scale=False):
    """log_cols: log(1+x)할 칸 (주차에 0이 있어 log1p로 통일)
    scale: 표준화 여부. OLS는 예측이 같아 기본 끔. Ridge 등을 쓰게 되면 켬 (그때는 모든 수치형에 적용)"""
    scaler = StandardScaler() if scale else "passthrough"
    log_cols = list(log_cols)
    log1p = FunctionTransformer(np.log1p, feature_names_out="one-to-one")
    return ColumnTransformer([
        ("num_log", Pipeline([("log", log1p), ("scale", StandardScaler() if scale else "passthrough")]), log_cols),
        ("num", scaler, [c for c in NUM if c not in log_cols]),
        ("cat", _cat(OneHotEncoder(drop=BASE, handle_unknown="ignore")), CAT),  # 학습 때 없던 동 → 기준(화곡동)과 같은 0
        ("flag", "passthrough", FLAGS),
    ], verbose_feature_names_out=False)


def tree_prep():
    """트리 계열은 값의 순서만 보므로 log·스케일링이 결과를 바꾸지 않음 → 수치형 그대로. 범주는 번호로(라벨 인코딩)"""
    return ColumnTransformer([
        ("num", "passthrough", NUM),
        ("cat", _cat(OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)), CAT),  # 모르는 동 → -1
        ("flag", "passthrough", FLAGS),
    ], verbose_feature_names_out=False)


def log_target(model):
    """타깃을 log(거래가)로 학습 → predict()는 exp로 되돌린 원 단위 값을 냄. 거래가는 모두 양수라 log 그대로 사용"""
    return TransformedTargetRegressor(regressor=model, func=np.log, inverse_func=np.exp)
