"""7단계 튜닝 — Optuna(베이지안 최적화), 학습용만, 5단계와 같은 지번 단위 5조각 (결정기록.md 12번)

대상: ElasticNet, LightGBM, XGBoost, MLP, CatBoost (MLP 30회, 나머지 50회, 빠른 모델부터)
  ElasticNet은 5단계에서 ElasticNetCV(지번 구분 없는 내부 교차검증)로 자동 선택 → 지번 단위 조각으로 다시 튜닝 (사용자 결정)
제외: OLS(튜닝할 설정값 없음), TabPFN(CPU에서 너무 느려 사용자 결정으로 제외)
목표: 교차검증 MAPE(원 단위) 최소화. 첫 시도는 기본 설정으로 시작(튜닝 결과가 기본보다 나빠지지 않게)
가지치기: 조각을 하나씩 끝낼 때마다 그때까지의 MAPE가 다른 시도들의 중간값보다 나쁘면 그 시도를 중단
주의: 튜닝과 평가에 같은 조각을 쓰므로 점수가 약간 좋게 나옴 → 최종 성능은 9단계 시험용으로 확인
이어 하기: 시도 기록을 reports/optuna.db에 저장 → 중간에 멈춰도 다시 실행하면 멈춘 곳부터 이어서 진행 (10/13 메모리 부족 중단 뒤 추가)
학습 곡선 (리포트 6번, 10/13 사용자 요청): 튜닝된 설정으로 5조각을 다시 학습하며 학습·검증 MAPE 기록 → 과소·과대적합 확인
출력: reports/best_params.json, reports/tuning_results.csv, reports/tuning_trials.csv(모든 시도 기록), reports/tuning_report.html
실행: python src/tune_models.py            (본 실행, reports/optuna.db)
      python src/tune_models.py 2          (시험: 모든 모델 2회씩, reports/optuna_test.db)
"""
import base64
import io
import json
import sys
import time
import warnings
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

from prep import TARGET, log_target, regression_prep, tree_prep

ROOT = Path(__file__).resolve().parent.parent
PROC, REPORT = ROOT / "data" / "processed", ROOT / "reports"
RECENT = "2025-10-01"
ORDER = ["ElasticNet", "LightGBM", "XGBoost", "MLP", "CatBoost"]
N_ALL = int(sys.argv[1]) if __name__ == "__main__" and len(sys.argv) > 1 else None  # 숫자를 주면 시험 실행 (모든 모델 같은 횟수, 별도 기록 파일). 직접 실행할 때만 읽음 — predict.py가 import할 때 --input을 숫자로 읽지 않게
N_TRIALS = {n: N_ALL or (30 if n == "MLP" else 50) for n in ORDER}  # MLP는 가장 느려서 30회 (10/13 사용자 결정 B)
STORAGE = f"sqlite:///{(REPORT / ('optuna_test.db' if N_ALL else 'optuna.db')).as_posix()}"
FINISHED = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
COLOR = {"ElasticNet": "#66b2c2", "LightGBM": "#d1495b", "XGBoost": "#edae49", "MLP": "#2a9d8f", "CatBoost": "#8e6bbf"}
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.dpi": 110,
                     "axes.spines.top": False, "axes.spines.right": False})
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 탐색 범위: (설정값, 쉬운 뜻, 범위 설명) — 리포트 표에도 그대로 씀
SPACE_DOC = {
    "ElasticNet": [("alpha", "규제 세기", "0.00001~1 (로그 간격)"), ("l1_ratio", "라쏘 비율 (0에 가까우면 릿지, 1이면 라쏘)", "0.05~1")],
    "LightGBM": [("n_estimators", "나무 개수", "100~1,500"), ("learning_rate", "한 번에 고치는 정도", "0.01~0.3 (로그 간격)"),
                 ("num_leaves", "나무 하나의 잎 개수", "15~255 (로그 간격)"), ("min_child_samples", "잎 하나에 필요한 최소 거래 수", "5~100"),
                 ("subsample", "나무마다 쓰는 행 비율", "0.5~1"), ("colsample_bytree", "나무마다 쓰는 칸 비율", "0.5~1"),
                 ("reg_lambda", "규제 세기", "0.001~10 (로그 간격)")],
    "XGBoost": [("n_estimators", "나무 개수", "100~1,500"), ("learning_rate", "한 번에 고치는 정도", "0.01~0.3 (로그 간격)"),
                ("max_depth", "나무 깊이", "3~10"), ("min_child_weight", "잎 하나에 필요한 최소 무게", "1~20 (로그 간격)"),
                ("subsample", "나무마다 쓰는 행 비율", "0.5~1"), ("colsample_bytree", "나무마다 쓰는 칸 비율", "0.5~1"),
                ("reg_lambda", "규제 세기", "0.001~10 (로그 간격)")],
    "MLP": [("n_layers", "은닉층 수", "1~3"), ("units_i", "층마다 신경 단위 수", "32~128 (로그 간격)"),
            ("alpha", "규제 세기", "0.00001~0.1 (로그 간격)"), ("learning_rate_init", "학습률", "0.0001~0.01 (로그 간격)"),
            ("batch_size", "한 번에 보는 거래 수", "64 / 128 / 200 / 256")],
    "CatBoost": [("iterations", "나무 개수", "300~1,500"), ("depth", "나무 깊이", "4~8"),
                 ("learning_rate", "한 번에 고치는 정도", "0.01~0.3 (로그 간격)"), ("l2_leaf_reg", "규제 세기", "1~10 (로그 간격)"),
                 ("bagging_temperature", "행을 무작위로 섞는 정도", "0~1")],
}
DEFAULTS = {  # 첫 시도 = 각 라이브러리 기본값 (범위 안으로 맞춤)
    "ElasticNet": {"alpha": 0.000684, "l1_ratio": 0.5},  # 5단계 ElasticNetCV가 학습용 전체에서 고른 값
    "LightGBM": {"n_estimators": 100, "learning_rate": 0.1, "num_leaves": 31, "min_child_samples": 20, "subsample": 1.0,
                 "colsample_bytree": 1.0, "reg_lambda": 0.001},
    "XGBoost": {"n_estimators": 100, "learning_rate": 0.3, "max_depth": 6, "min_child_weight": 1.0, "subsample": 1.0,
                "colsample_bytree": 1.0, "reg_lambda": 1.0},
    "MLP": {"n_layers": 1, "units_0": 100, "alpha": 0.0001, "learning_rate_init": 0.001, "batch_size": 200},
    "CatBoost": {"iterations": 1000, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3.0, "bagging_temperature": 1.0},
}


def suggest(trial, name):
    s = trial.suggest_int, trial.suggest_float
    if name == "ElasticNet":
        return {"alpha": s[1]("alpha", 1e-5, 1.0, log=True), "l1_ratio": s[1]("l1_ratio", .05, 1.0)}
    if name == "LightGBM":
        return {"n_estimators": s[0]("n_estimators", 100, 1500), "learning_rate": s[1]("learning_rate", .01, .3, log=True),
                "num_leaves": s[0]("num_leaves", 15, 255, log=True), "min_child_samples": s[0]("min_child_samples", 5, 100),
                "subsample": s[1]("subsample", .5, 1.0), "colsample_bytree": s[1]("colsample_bytree", .5, 1.0),
                "reg_lambda": s[1]("reg_lambda", 1e-3, 10, log=True)}
    if name == "XGBoost":
        return {"n_estimators": s[0]("n_estimators", 100, 1500), "learning_rate": s[1]("learning_rate", .01, .3, log=True),
                "max_depth": s[0]("max_depth", 3, 10), "min_child_weight": s[1]("min_child_weight", 1, 20, log=True),
                "subsample": s[1]("subsample", .5, 1.0), "colsample_bytree": s[1]("colsample_bytree", .5, 1.0),
                "reg_lambda": s[1]("reg_lambda", 1e-3, 10, log=True)}
    if name == "MLP":
        p = {"n_layers": s[0]("n_layers", 1, 3)}
        p.update({f"units_{i}": s[0](f"units_{i}", 32, 128, log=True) for i in range(p["n_layers"])})
        p.update({"alpha": s[1]("alpha", 1e-5, .1, log=True), "learning_rate_init": s[1]("learning_rate_init", 1e-4, 1e-2, log=True),
                  "batch_size": trial.suggest_categorical("batch_size", [64, 128, 200, 256])})
        return p
    return {"iterations": s[0]("iterations", 300, 1500), "depth": s[0]("depth", 4, 8), "learning_rate": s[1]("learning_rate", .01, .3, log=True),
            "l2_leaf_reg": s[1]("l2_leaf_reg", 1, 10, log=True), "bagging_temperature": s[1]("bagging_temperature", 0, 1)}


def build(name, p):
    if name == "ElasticNet":  # 회귀용 31칸 + 표준화 (규제가 계수 크기에 걸리므로 필수)
        return log_target(make_pipeline(regression_prep(scale=True), ElasticNet(max_iter=10000, **p)))
    if name == "ElasticNetCV":  # 5단계 방식 그대로 (튜닝 전 점수 계산용)
        return log_target(make_pipeline(regression_prep(scale=True), ElasticNetCV(l1_ratio=[.1, .5, .9, 1.0], cv=5, random_state=42, n_jobs=-1)))
    if name == "LightGBM":
        return log_target(make_pipeline(tree_prep(), LGBMRegressor(random_state=42, verbose=-1, subsample_freq=1, **p)))
    if name == "XGBoost":
        return log_target(make_pipeline(tree_prep(), XGBRegressor(random_state=42, **p)))
    if name == "MLP":
        layers = tuple(p[f"units_{i}"] for i in range(p["n_layers"]))
        mlp = MLPRegressor(hidden_layer_sizes=layers, alpha=p["alpha"], learning_rate_init=p["learning_rate_init"],
                           batch_size=p["batch_size"], early_stopping=True, max_iter=1000, random_state=42)
        return log_target(make_pipeline(regression_prep(scale=True), mlp))
    return log_target(make_pipeline(tree_prep(), CatBoostRegressor(random_seed=42, verbose=0, allow_writing_files=False, **p)))  # 학습 일지 폴더(catboost_info) 안 만듦


def mape(t, p):
    return float(np.mean(np.abs(p - t) / t) * 100)


def metrics(t, p):
    ape = np.abs(p - t) / t
    return {"MAPE(%)": ape.mean() * 100, "중앙값 오차율(%)": np.median(ape) * 100, "±10% 안 비율(%)": (ape <= .1).mean() * 100}


def cv_oof(name, p, trial=None):
    """5조각 교차검증 예측. 튜닝 중이면 조각마다 중간 점수를 Optuna에 알리고, 가망 없으면 중단"""
    oof = np.full(len(df), np.nan)
    for k, (tr_i, va_i) in enumerate(FOLDS):
        oof[va_i] = build(name, p).fit(df.iloc[tr_i], y[tr_i]).predict(df.iloc[va_i])
        if trial is not None:
            done = ~np.isnan(oof)
            trial.report(mape(y[done], oof[done]), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return oof


ITER_KEY = {"LightGBM": "n_estimators", "XGBoost": "n_estimators", "CatBoost": "iterations"}


class _RecordingMLP(MLPRegressor):
    """학습 곡선용 MLP: 튜닝 모델과 똑같이 학습하면서 epoch이 끝날 때마다 학습·검증 조각 예측(log)을 record_에 기록
    _update_no_improvement_count = scikit-learn 1.7.2 내부에서 epoch 끝마다 불리는 함수 (버전이 바뀌면 이름 확인 필요)"""

    def _update_no_improvement_count(self, early_stopping, X, y, sample_weight):
        super()._update_no_improvement_count(early_stopping, X, y, sample_weight)
        self.record_.append((self.predict(self.watch_[0]), self.predict(self.watch_[1])))


def _staged(name, p, xs, Xtr, ytr, Xva):
    """학습 곡선용: xs의 각 지점에서 (학습 조각 예측, 검증 조각 예측)을 원 단위로 차례로 냄"""
    if name == "ElasticNet":  # 지점 = 규제 세기 alpha. 지점마다 새로 학습
        for a in xs:
            m = build(name, {**p, "alpha": a}).fit(Xtr, ytr)
            yield m.predict(Xtr), m.predict(Xva)
    elif name == "MLP":  # 지점 = epoch. 튜닝 모델과 같은 설정·난수·조기 종료용 10% 분리로 학습하되, 멈추지 않고 끝 epoch까지
        prep = regression_prep(scale=True).fit(Xtr)
        A, B = prep.transform(Xtr), prep.transform(Xva)
        net = _RecordingMLP(hidden_layer_sizes=tuple(p[f"units_{i}"] for i in range(p["n_layers"])), alpha=p["alpha"],
                            learning_rate_init=p["learning_rate_init"], batch_size=p["batch_size"], early_stopping=True,
                            max_iter=len(xs), n_iter_no_change=len(xs) + 1, random_state=42)  # 멈춤 조건만 끔 → 학습 경로는 튜닝 모델과 같음
        net.watch_, net.record_ = (A, B), []
        net.fit(A, np.log(ytr))
        for ptr, pva in net.record_:
            yield np.exp(ptr), np.exp(pva)
    else:  # 트리: 끝 지점까지 한 번 학습한 뒤, 앞에서부터 나무 k개만 써서 예측
        m = build(name, {**p, ITER_KEY[name]: int(xs[-1])}).fit(Xtr, ytr).regressor_
        A, B, est = m[:-1].transform(Xtr), m[:-1].transform(Xva), m[-1]
        for k in xs:
            kw = {"LightGBM": {"num_iteration": k}, "XGBoost": {"iteration_range": (0, k)}, "CatBoost": {"ntree_end": k}}[name]
            yield np.exp(est.predict(A, **kw)), np.exp(est.predict(B, **kw))


def learning_curve(name, p):
    """학습 곡선 (과소·과대적합 확인, 10/13 사용자 요청): 튜닝된 설정으로 5조각을 다시 학습하며 학습 조각·검증 조각 MAPE 기록
    가로축 = 트리는 나무 개수, MLP는 epoch, 튜닝값의 2배까지 (사용자 결정). 검증 조각이 그림의 'Test loss' 자리 (시험용은 안 씀)
    ElasticNet은 학습 시간 개념이 없어 규제 세기(alpha)를 가로축으로 한 검증 곡선 (사용자 결정)
    검증 MAPE = 5조각 예측을 모아 한 번에 계산(튜닝 MAPE와 같은 방식), 학습 MAPE = 조각별 학습 MAPE 평균
    MLP는 튜닝값이 고정 숫자가 아니라 조기 종료 → 조각마다 고른 epoch이 다름. 튜닝값 = 조각별로 고른 epoch 목록,
      튜닝값의 MAPE = 조각마다 자기가 고른 epoch의 예측을 모아 계산 (실제 튜닝 모델 예측과 같은지 확인)
    반환: 가로축 지점, 학습 MAPE, 검증 MAPE, 튜닝값, 튜닝값에서의 (학습 MAPE, 검증 MAPE), 조각별 검증 MAPE"""
    if name == "ElasticNet":
        mark = p["alpha"]
        xs = np.unique(np.r_[np.logspace(-5, 0, 16), mark])
    elif name == "MLP":  # 실제 튜닝 모델(조기 종료) → 조각마다 검증 점수가 가장 좋았던 epoch = 되돌아간 epoch
        tuned = [build(name, p).fit(df.iloc[t], y[t]) for t, _ in FOLDS]
        mark = [int(np.argmax(m.regressor_[-1].validation_scores_)) + 1 for m in tuned]
        xs = np.arange(1, 2 * max(mark) + 1)
    else:
        mark = p[ITER_KEY[name]]
        xs = np.unique(np.r_[np.linspace(10, 2 * mark, 40).astype(int), mark])
    tr, va, fold_va = np.zeros((5, len(xs))), np.full((len(xs), len(df)), np.nan), np.zeros((5, len(xs)))
    for k, (tr_i, va_i) in enumerate(FOLDS):
        for j, (ptr, pva) in enumerate(_staged(name, p, xs, df.iloc[tr_i], y[tr_i], df.iloc[va_i])):
            tr[k, j], va[j, va_i], fold_va[k, j] = mape(y[tr_i], ptr), pva, mape(y[va_i], pva)
    va_mape = np.array([mape(y, v) for v in va])
    if name != "MLP":
        i = int(np.where(xs == mark)[0][0])
        return xs, tr.mean(0), va_mape, mark, (tr.mean(0)[i], va_mape[i]), fold_va
    oof = np.full(len(df), np.nan)
    for (_, va_i), b, m in zip(FOLDS, mark, tuned):
        oof[va_i] = va[b - 1, va_i]
        assert np.allclose(oof[va_i], m.predict(df.iloc[va_i])), "MLP 기록 경로가 실제 튜닝 모델과 다름"
    return xs, tr.mean(0), va_mape, mark, (np.mean([tr[k, b - 1] for k, b in enumerate(mark)]), mape(y, oof)), fold_va


def curve_img(name, xs, tr, va, mark, at_mark, fold_va):
    """학습 곡선 그림. 배경은 검증 MAPE 최저 지점 기준 — 덜 복잡한 쪽 파랑(과소적합 쪽), 더 복잡한 쪽 주황(과대적합 쪽)
    튜닝값: 숫자면 점선. 조각별 목록(MLP 조기 종료)이면 회색 구간 + 조각별 검증 곡선(가는 선)과 고른 지점(점)
    → 들쭉날쭉한 각 조각 곡선의 골짜기를 골라서 튜닝 결과가 5조각 평균 곡선보다 낮다는 것이 그림으로 보임"""
    en = name == "ElasticNet"  # 규제가 셀수록 단순한 모델 → 가로축을 뒤집어 오른쪽이 복잡한 쪽이 되게
    i_best = int(np.argmin(va))
    regions = [((xs[i_best], xs.max()) if en else (xs.min(), xs[i_best]), "#e8f0fa", "#4a6fa5", "과소적합 쪽"),
               ((xs.min(), xs[i_best]) if en else (xs[i_best], xs.max()), "#fbeee4", "#c26a2d", "과대적합 쪽")]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    if en:
        ax.set_xscale("log")
        ax.invert_xaxis()
        # 기본 눈금(10의 거듭제곱)은 수학 글꼴의 빼기 기호가 한글 글꼴에 없어 네모로 깨짐 → 일반 숫자로 표시
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    span = (lambda a, b: np.log(b / a) / np.log(xs.max() / xs.min())) if en else (lambda a, b: (b - a) / (xs.max() - xs.min()))
    for (a, b), fill, ink, label in regions:
        ax.axvspan(a, b, color=fill, lw=0)
        if span(a, b) > .12:  # 구역이 너무 좁으면 글자 생략
            ax.text(np.sqrt(a * b) if en else (a + b) / 2, .96, label, transform=ax.get_xaxis_transform(), ha="center", va="top", color=ink, fontsize=10)
    ax.plot(xs, tr, color="#222", lw=1.8, label="학습 조각 MAPE (본 데이터)")
    ax.plot(xs, va, color="#d62828", lw=1.8, label="검증 조각 MAPE (처음 보는 데이터)")
    if isinstance(mark, list):
        ax.axvspan(min(mark), max(mark), color="#888", alpha=.25, lw=0)
        for k, b in enumerate(mark):
            ax.plot(xs, fold_va[k], color="#d62828", lw=.8, alpha=.3, label="검증 조각 MAPE (조각별)" if k == 0 else None)
            ax.scatter(b, fold_va[k, b - 1], color="#d62828", edgecolor="#000", s=40, zorder=5, label="조기 종료가 고른 지점 (조각별)" if k == 0 else None)
        ax.annotate(f"조기 종료가 고른 epoch: 조각별 {min(mark)}~{max(mark)} (점)\n튜닝 결과 검증 {at_mark[1]:.2f}% · 학습 {at_mark[0]:.2f}%",
                    (max(mark), at_mark[1]), xytext=(12, 24), textcoords="offset points", fontsize=8)
    else:
        ax.axvline(mark, color="#555", ls="--", lw=1)
        ax.annotate(f"튜닝값 {mark:g}\n검증 {at_mark[1]:.2f}% · 학습 {at_mark[0]:.2f}%", (mark, at_mark[1]),
                    xytext=(8, 14), textcoords="offset points", fontsize=8)
    ax.set_ylim(0, min(max(va.max(), tr.max()), va.min() * 3) * 1.05)  # 초반 큰 오차가 그림을 눌러 버리지 않게 위를 자름
    ax.set_xlabel({"ElasticNet": "규제 세기 alpha (로그 눈금, 오른쪽일수록 규제 약함 = 복잡한 모델)",
                   "MLP": "epoch (학습 조각을 한 바퀴 보는 단위)"}.get(name, "나무 개수 (학습 반복 횟수)"))
    ax.set_ylabel("MAPE(%)")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.25), ncol=2, fontsize=8, frameon=False)  # 그래프 밖 아래 → 글자 안 가림
    ax.set_title(f"{name} — {'검증 곡선' if en else '학습 곡선'} (5조각 평균)")
    return img(fig)


def img(fig):
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def table(d, index=False):
    return '<div class="tw">' + d.to_html(index=index, border=0, na_rep="—", escape=False) + "</div>"


# ── 데이터·조각 (compare_models.py와 같은 순서·같은 난수 → 같은 조각) ─────────────
df = pd.read_csv(PROC / "preprocessed_all.csv").merge(pd.read_csv(PROC / "split.csv"), on="id")
jibun = pd.read_csv(PROC / "input_all.csv", dtype=str)[["id", "jibun"]].astype({"id": int})
df = df[df.set == "train"].merge(jibun, on="id").reset_index(drop=True)
y = df[TARGET].to_numpy()
recent = (df.deal_date >= RECENT).to_numpy()
FOLDS = list(GroupKFold(n_splits=5, shuffle=True, random_state=42).split(df, groups=(df.gu + "|" + df.dong + "|" + df.jibun)))

if __name__ == "__main__":
    REPORT.mkdir(exist_ok=True)
    best_params, studies, rows, trials = {}, {}, [], []
    for name in ORDER:
        t0 = time.time()
        study = optuna.create_study(direction="minimize", study_name=name, storage=STORAGE, load_if_exists=True,
                                    sampler=optuna.samplers.TPESampler(seed=42),
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
        for t in study.trials:  # 지난 실행이 강제로 멈추며 남긴 '진행 중' 시도는 실패로 정리
            if t.state == optuna.trial.TrialState.RUNNING:
                study.tell(t.number, state=optuna.trial.TrialState.FAIL)
        if not study.trials:
            study.enqueue_trial(DEFAULTS[name])  # 첫 시도는 기본 설정
        left = N_TRIALS[name] - sum(t.state in FINISHED for t in study.trials)

        def log(st, tr, name=name):
            state = "중단" if tr.state == optuna.trial.TrialState.PRUNED else f"MAPE {tr.value:.2f}%"
            done = sum(t.state in FINISHED for t in st.trials)
            print(f"  {name} {done}/{N_TRIALS[name]}: {state} | 지금까지 최고 {st.best_value:.2f}%", flush=True)

        if left > 0:
            print(f"{name}: {N_TRIALS[name] - left}회는 지난 기록에서 불러옴, {left}회 진행", flush=True)
            study.optimize(lambda tr: mape(y, cv_oof(name, suggest(tr, name), tr)), n_trials=left, callbacks=[log])
        # 튜닝 전 = 진짜 기본 설정. CatBoost는 학습률을 라이브러리가 자동으로 정하므로 learning_rate를 빼고 계산 (5단계 값과 같아짐)
        default = {k: v for k, v in DEFAULTS[name].items() if not (name == "CatBoost" and k == "learning_rate")}
        # ElasticNet의 튜닝 전 = 5단계 방식(ElasticNetCV 자동 선택) → 17.23%와 같아야 함
        before = cv_oof("ElasticNetCV", {}) if name == "ElasticNet" else cv_oof(name, default)
        best, after = dict(study.best_params), cv_oof(name, study.best_params)
        if mape(y, after) > mape(y, before):  # 튜닝 결과가 기본 설정보다 나쁘면 기본 설정 유지
            best, after = (DEFAULTS[name] if name == "ElasticNet" else default), before
            print(f"{name}: 튜닝 결과가 기본 설정보다 나빠 기본 설정 유지", flush=True)
        studies[name], best_params[name] = study, best
        trials.append(study.trials_dataframe().assign(모델=name))
        for label, o in [("기본 설정", before), ("튜닝 후", after)]:
            rows.append({"모델": name, "구분": label, **metrics(y, o), **{f"최근12개월 {k}": v for k, v in metrics(y[recent], o[recent]).items()}})
        n_pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
        print(f"{name} 완료: 기본 {mape(y, before):.2f}% → 튜닝 {mape(y, after):.2f}% | 중단된 시도 {n_pruned}/{N_TRIALS[name]} | {(time.time() - t0) / 60:.1f}분", flush=True)
        (REPORT / "best_params.json").write_text(json.dumps(best_params, ensure_ascii=False, indent=2), encoding="utf-8")  # 모델마다 저장

    res = pd.DataFrame(rows)
    res.round(3).to_csv(REPORT / "tuning_results.csv", index=False, encoding="utf-8-sig")
    pd.concat(trials).to_csv(REPORT / "tuning_trials.csv", index=False, encoding="utf-8-sig")

    # ── 리포트 ─────────────────────────────────────────
    comp = pd.read_csv(REPORT / "model_compare.csv")  # 튜닝 안 한 모델(OLS·기준선)은 5단계 결과 그대로
    tuned = res[res.구분 == "튜닝 후"].drop(columns="구분").assign(상태="튜닝 후")
    others = comp[comp.모델.isin(["OLS", "기준선"])][list(tuned.columns.drop("상태"))].assign(상태="튜닝 안 함")
    final = pd.concat([tuned, others]).sort_values("MAPE(%)").reset_index(drop=True)
    pivot = res.pivot(index="모델", columns="구분", values="MAPE(%)").reindex(ORDER)
    pivot["개선(%p)"] = pivot["기본 설정"] - pivot["튜닝 후"]
    pivot = pivot.sort_values("튜닝 후")
    best = final.iloc[0]

    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = np.arange(len(pivot))
    ax.bar(x - .2, pivot["기본 설정"], .4, color="#bbb", label="기본 설정")
    ax.bar(x + .2, pivot["튜닝 후"], .4, color=[COLOR[n] for n in pivot.index], label="튜닝 후")
    for i, (b, a) in enumerate(zip(pivot["기본 설정"], pivot["튜닝 후"])):
        ax.text(i - .2, b, f"{b:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + .2, a, f"{a:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, pivot.index)
    ax.set_ylabel("MAPE(%)")
    ax.legend()
    ax.set_title("튜닝 전후 MAPE (낮을수록 좋음)")
    before_after = img(fig)

    fig, axes = plt.subplots(1, len(ORDER), figsize=(15, 3.6), sharey=False)
    for a, name in zip(axes, ORDER):
        tr = [t for t in studies[name].trials if t.state == optuna.trial.TrialState.COMPLETE]
        xs, vs = [t.number + 1 for t in tr], [t.value for t in tr]
        a.scatter(xs, vs, s=12, color=COLOR[name], alpha=.6)
        a.plot(xs, np.minimum.accumulate(vs), color="#333", lw=1.5)
        a.set_title(name)
        a.set_xlabel("시도 번호")
    axes[0].set_ylabel("MAPE(%)")
    history = img(fig)

    imp_html = ""
    for name in ORDER:
        try:
            imp = optuna.importance.get_param_importances(studies[name])
            imp_html += f"<h4>{name}</h4>" + table(pd.DataFrame({"설정값": list(imp), "영향 비율": [f"{v * 100:.0f}%" for v in imp.values()]}))
        except Exception as e:  # 끝까지 간 시도가 너무 적으면 계산 불가
            imp_html += f"<h4>{name}</h4><p class='muted'>계산 불가: {e}</p>"

    params_html = ""
    for name in ORDER:
        doc = {k: (m, r) for k, m, r in SPACE_DOC[name]}
        prow = []
        for k, v in best_params[name].items():
            key = "units_i" if k.startswith("units_") else k
            prow.append([f"<code>{k}</code>", doc.get(key, ("", ""))[0], doc.get(key, ("", ""))[1],
                         DEFAULTS[name].get(k, "—"), round(v, 5) if isinstance(v, float) else v])
        params_html += f"<h4>{name}</h4>" + table(pd.DataFrame(prow, columns=["설정값", "쉬운 뜻", "탐색 범위", "기본값", "최적값"]))

    # 학습 곡선 (리포트 6번)
    curve_html, crows = "", []
    for name in ORDER:
        t0 = time.time()
        xs, tr, va, mark, (tr_m, va_m), fold_va = learning_curve(name, best_params[name])
        en = name == "ElasticNet"
        i_best, i_end = int(np.argmin(va)), 0 if en else -1
        label = f"{min(mark)}~{max(mark)} (조각별)" if isinstance(mark, list) else f"{mark:g}"
        curve_html += curve_img(name, xs, tr, va, mark, (tr_m, va_m), fold_va)
        crows.append({"모델": name, "그래프": "검증 곡선" if en else "학습 곡선", "튜닝값": label,
                      "튜닝값 학습 MAPE(%)": tr_m, "튜닝값 검증 MAPE(%)": va_m, "간격(검증−학습, %p)": va_m - tr_m,
                      "검증 최저 지점": f"{xs[i_best]:g}", "검증 최저 MAPE(%)": va[i_best],
                      "가장 복잡한 끝 검증 MAPE(%)": va[i_end], "튜닝값 → 끝 변화(%p)": np.nan if isinstance(mark, list) else va[i_end] - va_m})  # MLP: 골짜기 값과 평균 곡선은 기준이 달라 빼지 않음
        tuned_mape = res[(res.모델 == name) & (res.구분 == "튜닝 후")]["MAPE(%)"].iloc[0]
        print(f"{name} 학습 곡선: 튜닝값 {label} 검증 {va_m:.3f}% (튜닝 결과 {tuned_mape:.3f}%) | {(time.time() - t0) / 60:.1f}분", flush=True)
    curves = pd.DataFrame(crows)

    css = (ROOT / "src" / "eda.py").read_text(encoding="utf-8").split("<style>")[1].split("</style>")[0].replace("{{", "{").replace("}}", "}")
    n_pruned = {n: sum(t.state == optuna.trial.TrialState.PRUNED for t in studies[n].trials) for n in ORDER}
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>튜닝 리포트</title><style>{css}</style></head><body><div class="wrap">
<header><h1>다세대 시세 모델 튜닝 리포트 (7단계)</h1>
<p>학습용 {len(df):,}건 · 지번 단위 5조각 교차검증 (5단계와 같은 조각) · Optuna {optuna.__version__} · 시도 {', '.join(f'{n} {v}회' for n, v in N_TRIALS.items())} · 생성 {date.today()} · <code>python src/tune_models.py</code></p></header>
<nav><a href="#s0">한눈에</a><a href="#s1">1. 튜닝 방법</a><a href="#s2">2. 튜닝 전후</a><a href="#s3">3. 전체 순위</a><a href="#s4">4. 최적 설정값</a><a href="#s5">5. 탐색 과정</a><a href="#s6">6. 학습 곡선</a><a href="#s7">7. 주의</a></nav>
<section id="s0"><h2>한눈에 보기</h2><ol>
<li><b>가장 낮은 MAPE: {best.모델}</b> {best['MAPE(%)']:.2f}% ({best.상태}) · 최근 12개월 {best['최근12개월 MAPE(%)']:.2f}% · ±10% 안 {best['±10% 안 비율(%)']:.1f}%</li>
<li><b>튜닝으로 가장 많이 좋아진 모델:</b> {pivot['개선(%p)'].idxmax()} ({pivot['개선(%p)'].max():.2f}%p 개선)</li>
<li>모델별 개선: {", ".join(f"{n} {r['기본 설정']:.2f} → {r['튜닝 후']:.2f}%" for n, r in pivot.iterrows())}</li></ol></section>
<section id="s1"><h2>1. 튜닝 방법</h2>{table(pd.DataFrame([
    ["도구", f"Optuna {optuna.__version__} — 베이지안 최적화(TPE): 앞선 시도 결과를 보고 좋아 보이는 설정값 쪽을 골라 다음 시도"],
    ["목표", "지번 단위 5조각 교차검증 MAPE(원 단위) 최소화 — 5단계 비교와 같은 조각·같은 지표"],
    ["시도 횟수", f"{', '.join(f'{n} {v}회' for n, v in N_TRIALS.items())} (MLP는 가장 느려 줄임). 첫 시도는 기본 설정 → 튜닝 결과가 기본보다 나빠지지 않음. 기록은 reports/optuna.db에 저장돼 중단돼도 이어서 진행"],
    ["가지치기", "조각 하나가 끝날 때마다 그때까지의 MAPE가 다른 시도들의 중간값보다 나쁘면 중단 (처음 5회는 끝까지). 중단된 시도: " + ", ".join(f"{n} {v}회" for n, v in n_pruned.items())],
    ["튜닝 대상", "ElasticNet, LightGBM, XGBoost, MLP, CatBoost. ElasticNet의 '기본 설정' = 5단계 ElasticNetCV 자동 선택(지번 구분 없는 내부 교차검증) → 지번 단위 조각으로 다시 튜닝"],
    ["튜닝 안 함", "OLS(설정값 없음), TabPFN(CPU에서 너무 느려 제외)"],
], columns=["항목", "내용"]))}</section>
<section id="s2"><h2>2. 튜닝 전후 MAPE</h2>{before_after}{table(pivot.round(2), index=True)}</section>
<section id="s3"><h2>3. 전체 순위 (튜닝 후 + 튜닝 안 한 모델)</h2>{table(final.round(2))}</section>
<section id="s4"><h2>4. 최적 설정값</h2>{params_html}<h3>설정값별 영향 비율</h3><p class="why">그 설정값을 바꿨을 때 MAPE가 얼마나 달라졌는지의 상대 비율 (Optuna 계산). 높을수록 튜닝할 가치가 큰 설정값.</p>{imp_html}</section>
<section id="s5"><h2>5. 탐색 과정</h2><p class="why">점 = 끝까지 간 시도의 MAPE, 선 = 그때까지의 최고 기록. 선이 일찍 평평해지면 더 시도해도 나아질 여지가 적다는 뜻.</p>{history}</section>
<section id="s6"><h2>6. 학습 곡선 — 과소·과대적합 확인</h2>
<p class="why">검은 선 = 학습 조각(모델이 본 데이터) MAPE, 빨간 선 = 검증 조각(처음 보는 데이터) MAPE. <b>두 선이 모두 높으면 과소적합</b>(덜 배움), <b>학습 선은 계속 내려가는데 검증 선이 바닥을 찍고 다시 오르면 과대적합</b>(외움). 두 선 사이 간격은 있어도 정상 — 판단 기준은 검증 선이 다시 오르는지. 배경은 검증 MAPE가 가장 낮은 지점 기준, 점선 = 튜닝값 (MLP는 회색 구간과 점). 가로축은 튜닝값의 2배까지 (ElasticNet은 학습 시간 개념이 없어 규제 세기 0.00001~1의 검증 곡선). 검증 조각이 일반적인 그림의 'Test loss' 자리 — 시험용 2,200건은 쓰지 않음.</p>
{table(curves.round(2))}
<p class="muted">튜닝값 → 끝 변화: 튜닝값에서 가장 복잡한 끝(트리·MLP = 튜닝값의 2배, ElasticNet = 규제 가장 약함)까지 검증 MAPE가 몇 %p 변했나. 양수 = 더 복잡하게 하면 나빠짐(과대적합 시작), 음수 = 더 좋아짐(아직 덜 배움). MLP는 —: 조기 종료가 고른 골짜기 값과 5조각 평균 곡선은 기준이 달라 빼지 않음.</p>
{curve_html}
<p class="muted">MLP는 튜닝값이 고정된 숫자가 아니라 <b>조기 종료</b>: 학습 조각의 10%를 연습용으로 떼고, 연습용 점수가 10번 연속 안 오르면 멈춘 뒤 가장 좋았던 epoch으로 되돌아감 → 조각마다 고른 epoch이 달라 회색 구간(48~76 등)으로 표시. 곡선은 튜닝 모델과 같은 조건으로 학습하되 멈추지 않고 끝까지 간 경로. 가는 선 = 조각별 검증 곡선, 점 = 각 조각이 고른 지점. 성적이 epoch마다 들쭉날쭉해서 조기 종료는 각 조각 곡선의 골짜기를 고름 → 점들을 모은 튜닝 결과가 5조각 평균 곡선(굵은 빨간 선)보다 낮게 나옴. 모든 모델에서 표의 튜닝값 검증 MAPE = 튜닝 결과 (실행 중 확인). MLP의 학습 MAPE에는 연습용 10%도 포함.</p></section>
<section id="s7"><h2>7. 주의</h2><ul>
<li>튜닝과 평가에 <b>같은 5조각</b>을 썼기 때문에 튜닝 후 점수는 약간 낙관적임 → 최종 성능은 9단계에서 시험용 2,200건으로 한 번만 측정</li>
<li>개선 폭이 작으면 튜닝 효과보다 조각 간 흔들림(5단계 표준편차 0.7~2.4%p) 안의 차이일 수 있음</li></ul></section>
</div></body></html>"""
    (REPORT / "tuning_report.html").write_text(html, encoding="utf-8")
    print(final.round(2).to_string(index=False))
    print("저장: reports/best_params.json, reports/tuning_results.csv, reports/tuning_report.html")
