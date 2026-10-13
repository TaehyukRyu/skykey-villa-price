"""5단계 1차 모델 비교 — 학습용만, 지번 단위 5조각 교차검증 (결정기록.md 11번)

후보: 기준선, OLS, ElasticNet, LightGBM, XGBoost, CatBoost, MLP(딥러닝) (모두 기본 설정, 난수 42)
입력: 회귀 = prep.regression_prep (OLS 스케일 끔, ElasticNet·MLP 켬), 트리 = prep.tree_prep
타깃: log(거래가)로 학습 → 예측은 원 단위로 되돌려 오차 계산
시험용(test)은 쓰지 않음 (9단계에서 한 번만)
출력: reports/model_compare.csv, reports/model_report.html
실행: python src/compare_models.py
"""
import base64
import io
import time
import warnings
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

from prep import LOG_COLS, TARGET, log_target, regression_prep, tree_prep

ROOT = Path(__file__).resolve().parent.parent
PROC, REPORT = ROOT / "data" / "processed", ROOT / "reports"
RECENT = "2025-10-01"  # 회사 검증 물건 = 최근 12개월 거래 → 이 날짜 이후 거래만 따로도 측정
GUS = ["강남구", "강서구", "관악구"]
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.dpi": 110,
                     "axes.spines.top": False, "axes.spines.right": False})
warnings.filterwarnings("ignore", category=UserWarning)

MODELS = {  # 이름: (계열, 만드는 함수)
    "OLS": ("회귀", lambda: log_target(make_pipeline(regression_prep(), LinearRegression()))),
    "ElasticNet": ("회귀", lambda: log_target(make_pipeline(
        regression_prep(scale=True), ElasticNetCV(l1_ratio=[.1, .5, .9, 1.0], cv=5, random_state=42, n_jobs=-1)))),
    "LightGBM": ("트리", lambda: log_target(make_pipeline(tree_prep(), LGBMRegressor(random_state=42, verbose=-1)))),
    "XGBoost": ("트리", lambda: log_target(make_pipeline(tree_prep(), XGBRegressor(random_state=42)))),
    "CatBoost": ("트리", lambda: log_target(make_pipeline(tree_prep(), CatBoostRegressor(random_seed=42, verbose=0, allow_writing_files=False)))),
    # 딥러닝 기준: 기본 신경망(은닉층 100개 1층). 신경망은 스케일링 필수 → 표준화 켬. 조기 종료로 과적합·시간 줄임
    "MLP": ("딥러닝", lambda: log_target(make_pipeline(
        regression_prep(scale=True), MLPRegressor(early_stopping=True, max_iter=1000, random_state=42)))),
}
COLOR = {"기준선": "#9e9e9e", "OLS": "#00798c", "ElasticNet": "#66b2c2", "LightGBM": "#d1495b", "XGBoost": "#edae49",
         "CatBoost": "#8e6bbf", "MLP": "#2a9d8f"}


def metrics(y, p):
    ape = np.abs(p - y) / y  # 오차율 = |예측 − 실제| ÷ 실제
    return {"MAPE(%)": ape.mean() * 100, "중앙값 오차율(%)": np.median(ape) * 100, "±10% 안 비율(%)": (ape <= .1).mean() * 100}


def img(fig):
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def table(d, index=False):
    return '<div class="tw">' + d.to_html(index=index, border=0, na_rep="—", escape=False) + "</div>"


# ── 데이터 ──────────────────────────────────────────────
df = pd.read_csv(PROC / "preprocessed_all.csv").merge(pd.read_csv(PROC / "split.csv"), on="id")
jibun = pd.read_csv(PROC / "input_all.csv", dtype=str)[["id", "jibun"]].astype({"id": int})
df = df[df.set == "train"].merge(jibun, on="id").reset_index(drop=True)
groups = (df.gu + "|" + df.dong + "|" + df.jibun).to_numpy()
y = df[TARGET].to_numpy()
recent = (df.deal_date >= RECENT).to_numpy()

# ── 교차검증: 각 행은 자기가 검증 조각일 때 나온 예측(OOF)만 가짐 ──────────
gkf = GroupKFold(n_splits=5, shuffle=True, random_state=42)
names = ["기준선"] + list(MODELS)
oof = {n: np.full(len(df), np.nan) for n in names}
fold = np.zeros(len(df), int)
sec = {n: 0.0 for n in names}
for k, (tr_i, va_i) in enumerate(gkf.split(df, groups=groups)):
    tr, va = df.iloc[tr_i], df.iloc[va_i]
    fold[va_i] = k
    assert not set(groups[tr_i]) & set(groups[va_i])  # 같은 지번이 학습·검증 조각에 동시에 없음
    t0 = time.time()
    ppm = tr[TARGET] / tr.area_m2  # 기준선: 동별 ㎡당 중앙값 × 면적 (프로세스 파일 5단계)
    dong_med, gu_med = ppm.groupby(tr.dong).median(), ppm.groupby(tr.gu).median()
    base_ppm = va.dong.map(dong_med).fillna(va.gu.map(gu_med))  # 학습 조각에 없는 동(예: 세곡동 3건) → 같은 구 중앙값
    oof["기준선"][va_i] = base_ppm.to_numpy() * va.area_m2.to_numpy()
    sec["기준선"] += time.time() - t0
    for name, (_, make) in MODELS.items():
        t0 = time.time()
        oof[name][va_i] = make().fit(tr, tr[TARGET]).predict(va)
        sec[name] += time.time() - t0
    print(f"조각 {k + 1}/5 완료 (학습 {len(tr_i):,} / 검증 {len(va_i):,})", flush=True)

rows = []
for n in names:
    fm = [metrics(y[fold == k], oof[n][fold == k])["MAPE(%)"] for k in range(5)]
    rows.append({"모델": n, "계열": "기준" if n == "기준선" else MODELS[n][0], **metrics(y, oof[n]),
                 **{f"최근12개월 {k}": v for k, v in metrics(y[recent], oof[n][recent]).items()},
                 "조각별 MAPE 표준편차": np.std(fm), "조각당 학습시간(초)": sec[n] / 5})
res = pd.DataFrame(rows).sort_values("MAPE(%)").reset_index(drop=True)
REPORT.mkdir(exist_ok=True)
res.round(3).to_csv(REPORT / "model_compare.csv", index=False, encoding="utf-8-sig")
best = res.iloc[0].모델
best_reg = res[res.계열 == "회귀"].iloc[0].모델
best_tree = res[res.계열 == "트리"].iloc[0].모델
base = res.set_index("모델").loc["기준선", "MAPE(%)"]

# ── 그래프 ────────────────────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
for a, col, title in [(ax[0], "MAPE(%)", "MAPE (낮을수록 좋음)"), (ax[1], "±10% 안 비율(%)", "±10% 안에 든 비율 (높을수록 좋음)")]:
    x = np.arange(len(res))
    a.bar(x - .2, res[col], .4, color=[COLOR[n] for n in res.모델], label="전체")
    a.bar(x + .2, res[f"최근12개월 {col}"], .4, color=[COLOR[n] for n in res.모델], alpha=.45, hatch="//", label="최근 12개월")
    for i, v in enumerate(res[col]):
        a.text(i - .2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    a.set_xticks(x, res.모델)
    a.set_title(title)
    a.legend(fontsize=8)
bars = img(fig)

fig, ax = plt.subplots(figsize=(10, 3.6))
for i, n in enumerate(res.모델):
    fm = [metrics(y[fold == k], oof[n][fold == k])["MAPE(%)"] for k in range(5)]
    ax.scatter([i] * 5, fm, color=COLOR[n], s=40)
    ax.plot([i - .2, i + .2], [np.mean(fm)] * 2, color="#333", lw=2)
ax.set_xticks(range(len(res)), res.모델)
ax.set_ylabel("조각별 MAPE(%)")
ax.set_title("5개 검증 조각에서의 MAPE (점 = 조각, 가로선 = 평균) — 점이 모여 있을수록 안정적")
folds_fig = img(fig)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
cap = np.quantile(y, .99) / 1e8
for a, n in zip(ax, [best_reg, best_tree]):
    for g, c in zip(GUS, ["#d1495b", "#00798c", "#edae49"]):
        m = (df.gu == g).to_numpy()
        a.scatter(y[m] / 1e8, oof[n][m] / 1e8, s=4, alpha=.35, color=c, label=g)
    a.plot([0, cap], [0, cap], "k--", lw=1)
    a.plot([0, cap], [0, cap * 1.1], color="#999", lw=.8, ls=":")
    a.plot([0, cap], [0, cap * .9], color="#999", lw=.8, ls=":")
    a.set_xlim(0, cap); a.set_ylim(0, cap)
    a.set_xlabel("실제 거래가(억)"); a.set_ylabel("예측(억)")
    a.set_title(f"{n}: 예측 vs 실제 (점선 = ±10%)")
    a.legend(markerscale=4, fontsize=8)
scatter = img(fig)

fig, ax = plt.subplots(figsize=(10, 3.8))
for n in ["기준선", best_reg, best_tree]:
    ax.hist(np.clip((oof[n] / y - 1) * 100, -60, 60), bins=80, alpha=.45, color=COLOR[n], label=n, density=True)
ax.axvline(0, color="k", lw=.8)
ax.set_xlabel("예측 오차(%) = (예측 ÷ 실제 − 1) × 100   (±60% 밖은 끝에 모음)")
ax.legend()
err_hist = img(fig)

# ── 권역별 ────────────────────────────────────────────
gu_tbl = pd.DataFrame({n: [metrics(y[(df.gu == g).to_numpy()], oof[n][(df.gu == g).to_numpy()])["MAPE(%)"] for g in GUS] for n in res.모델},
                      index=[f"{g} ({int((df.gu == g).sum()):,}건)" for g in GUS]).round(1)

# ── 해석: OLS 계수 (statsmodels, 학습용 전체) / 최고 트리 모델 변수 중요도 ─────────
rp = regression_prep().fit(df)
X = pd.DataFrame(rp.transform(df), columns=rp.get_feature_names_out())
ols = sm.OLS(np.log(y), sm.add_constant(X)).fit()
coef = pd.DataFrame({"계수": ols.params, "p값": ols.pvalues}).drop("const")


def effect(name, b):
    if name in LOG_COLS:
        return f"이 값이 1% 늘면 거래가 {b:+.2f}%"
    if name.startswith(("dong_", "structure_")):
        return f"기준 범주 대비 {(np.exp(b) - 1) * 100:+.1f}%"
    if name.endswith("_missing"):
        return f"원래 빈칸이었으면 {(np.exp(b) - 1) * 100:+.1f}%"
    return f"1 늘면 거래가 {(np.exp(b) - 1) * 100:+.2f}%"


coef["해석"] = [effect(n, b) for n, b in coef.계수.items()]
coef["p값"] = coef.p값.map(lambda p: "<0.001" if p < .001 else f"{p:.3f}")
coef = coef.assign(계수=coef.계수.round(4)).reset_index(names="변수")

tree_name = best_tree
imp_model = {"LightGBM": LGBMRegressor(random_state=42, verbose=-1, importance_type="gain"),
             "XGBoost": XGBRegressor(random_state=42, importance_type="gain"),
             "CatBoost": CatBoostRegressor(random_seed=42, verbose=0, allow_writing_files=False)}[tree_name]
tm = log_target(make_pipeline(tree_prep(), imp_model)).fit(df, df[TARGET])
tp = tm.regressor_.steps[0][1]
imp = pd.Series(tm.regressor_.steps[-1][1].feature_importances_, index=tp.get_feature_names_out())
imp = (imp / imp.sum() * 100).sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(imp.index, imp.values, color=COLOR[tree_name])
ax.set_xlabel("중요도 비율(%)")
ax.set_title(f"{tree_name} 변수 중요도 (학습용 전체로 학습)")
imp_fig = img(fig)

# ── HTML ─────────────────────────────────────────────
show = res.copy()
show.insert(2, "기준선 대비", [f"{(v / base - 1) * 100:+.0f}%" if n != "기준선" else "—" for n, v in zip(show.모델, show["MAPE(%)"])])
show = show.round(2)
r_best = res.iloc[0]
findings = f"""<ol>
<li><b>가장 낮은 MAPE: {best}</b> {r_best['MAPE(%)']:.1f}% (기준선 {base:.1f}% 대비 {(r_best['MAPE(%)'] / base - 1) * 100:+.0f}%). 최근 12개월 거래만 보면 {r_best['최근12개월 MAPE(%)']:.1f}%</li>
<li><b>회귀 최고 {best_reg}</b> {res.set_index('모델').loc[best_reg, 'MAPE(%)']:.1f}% / <b>트리 최고 {best_tree}</b> {res.set_index('모델').loc[best_tree, 'MAPE(%)']:.1f}%</li>
<li><b>±10% 안에 든 비율:</b> {best} {r_best['±10% 안 비율(%)']:.1f}% — 10건 중 약 {r_best['±10% 안 비율(%)'] / 10:.0f}건이 실제 가격 ±10% 안</li>
<li><b>모든 모델이 기본 설정</b>이라 이 순위는 1차 비교. 튜닝은 7단계</li></ol>"""
css = (ROOT / "src" / "eda.py").read_text(encoding="utf-8").split("<style>")[1].split("</style>")[0].replace("{{", "{").replace("}}", "}")
html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>모델 비교 리포트</title><style>{css}</style></head><body><div class="wrap">
<header><h1>다세대 시세 모델 1차 비교 리포트</h1>
<p>학습용 {len(df):,}건 · 지번 단위 5조각 교차검증 · {len(names)}개 모델(기준선 포함) 기본 설정 · 생성 {date.today()} · <code>python src/compare_models.py</code></p></header>
<nav><a href="#s0">한눈에</a><a href="#s1">1. 비교 방법</a><a href="#s2">2. 평가 지표</a><a href="#s3">3. 결과</a><a href="#s4">4. 권역별</a><a href="#s5">5. 예측 vs 실제</a><a href="#s6">6. 모델 해석</a><a href="#s7">7. OLS vs ElasticNet</a><a href="#s8">8. 한계·다음 단계</a></nav>
<section id="s0"><h2>한눈에 보기</h2>{findings}</section>

<section id="s1"><h2>1. 비교 방법</h2>{table(pd.DataFrame([
    ["데이터", f"학습용 {len(df):,}건만 사용 (시험용 2,200건은 9단계에서 한 번만)"],
    ["검증 방식", "지번 단위 5조각 교차검증: 학습용을 5조각으로 나눠, 4조각으로 학습하고 1조각으로 검증하기를 5번 반복. 같은 건물(지번)은 한 조각에만 들어감 → 모델이 건물 가격을 외워서 맞히는 것을 막음"],
    ["타깃", "모든 모델이 log(거래가)로 학습 → 예측을 원 단위로 되돌려 오차 계산"],
    ["기준선", "동별 ㎡당 중앙값 × 전용면적 (학습 조각으로 중앙값 계산, 학습 조각에 없는 동은 같은 구 중앙값). 학습 없는 가장 단순한 방법 = 넘어야 할 최소 점수"],
    ["회귀 계열", "OLS(일반 선형회귀, 스케일 없음), ElasticNet(릿지+라쏘 규제, 스케일 켬, 규제 세기·비율은 학습 조각 안에서 자동 선택) — 입력 31칸"],
    ["트리 계열", "LightGBM, XGBoost, CatBoost (기본 설정) — 입력 16칸 (동·구조 라벨 인코딩)"],
    ["딥러닝", "MLP(scikit-learn 기본 신경망, 은닉층 100개 1층, 조기 종료) — 입력 31칸, 표준화 켬 (신경망은 스케일링 필수)"],
    ["최근 12개월", f"{RECENT} 이후 거래 {int(recent.sum()):,}건만 따로 계산 — 회사 검증 물건이 최근 12개월 거래에서 뽑히므로"],
], columns=["항목", "내용"]))}</section>

<section id="s2"><h2>2. 평가 지표</h2>
{table(pd.DataFrame([
    ["MAPE (평균 오차율)", "모든 거래의 |예측 − 실제| ÷ 실제 를 평균", "평균적으로 몇 % 틀리나", "크게 틀린 몇 건에 끌려 올라갈 수 있음"],
    ["중앙값 오차율", "같은 오차율의 중앙값", "보통의 물건은 몇 % 틀리나", "크게 틀린 건이 얼마나 심한지는 안 보임"],
    ["±10% 안 비율", "오차율 10% 이하인 거래 비율", "실무에서 '맞혔다'고 볼 만한 비율", "10%라는 선은 관례적 기준"],
], columns=["지표", "계산", "답하는 질문", "한계"]))}
<div class="callout"><b>왜 평균 오차율(MAPE)을 주 지표로 썼나</b><ul>
<li><b>과제가 오차율로 평가함:</b> 블라인드 검증에서 "실제 실거래가 대비 오차율"과 "하우스머치 AI 시세 대비 오차율"을 봄 (과제 07번)</li>
<li><b>가격대가 달라도 공정함:</b> 거래가가 0.2억 ~ 30억으로 폭이 넓음. 원 단위 오차(MAE, RMSE)는 1억 틀리는 게 2억 집이든 10억 집이든 같게 취급해 비싼 강남 거래가 점수를 좌우함. 오차율은 2억 집 2천만원 오차와 10억 집 1억 오차를 같은 10%로 봄</li>
<li><b>설명하기 쉬움:</b> "평균 12% 틀린다"는 비전공자도 바로 이해함 (과제 평가 '문서와 전달력')</li>
<li><b>타깃 log와 맞음:</b> log로 학습하면 모델이 원래 % 오차를 줄이도록 배움</li></ul>
MAPE 하나만 보면 크게 틀린 몇 건에 끌릴 수 있어서 <b>중앙값 오차율</b>과 <b>±10% 안 비율</b>을 함께 봄.</div>
<p class="why"><b>회귀 계열과 트리 계열의 평가 지표는 같음:</b> 지표는 모델 종류가 아니라 <b>문제 종류</b>로 정해짐. 여기서는 모두 "가격(숫자)을 맞히는 문제(회귀 문제)"라서
같은 지표로 비교함. 트리 모델도 여기서는 숫자를 내는 '트리 기반 회귀 모델'임. 정확도·F1 같은 지표는 "어느 범주인가"를 맞히는 분류 문제용.</p></section>

<section id="s3"><h2>3. 결과</h2>{bars}{table(show)}
<p class="muted">기준선 대비 = MAPE가 기준선보다 몇 % 줄었나. 조각별 MAPE 표준편차 = 5개 검증 조각 사이의 흔들림 (작을수록 안정적). 학습시간은 조각 1개 기준(초).</p>
{folds_fig}</section>

<section id="s4"><h2>4. 권역별 MAPE(%)</h2>{table(gu_tbl, index=True)}
<p class="muted">권역마다 오차 수준이 다르면 confidence(신뢰도)를 권역별로 다르게 매길 근거가 됨 (6단계).</p></section>

<section id="s5"><h2>5. 예측 vs 실제 · 오차 분포</h2>
<p class="why">대각선(검은 점선)에 붙을수록 정확. 회귀 최고({best_reg})와 트리 최고({best_tree})를 나란히 표시. 상위 1%({cap:.1f}억 초과)는 그림 밖.</p>
{scatter}{err_hist}
<p class="muted">0에 높고 좁게 몰릴수록 좋음. 오른쪽(+)으로 치우치면 비싸게, 왼쪽(−)이면 싸게 예측하는 경향.</p></section>

<section id="s6"><h2>6. 모델 해석</h2>
<h3>OLS 계수 (학습용 전체, 타깃 log) — basis 문장의 재료</h3>
<p class="why">타깃이 log라서 계수를 %로 읽음. log를 씌운 칸(pub_ppm 등)은 "1% 늘면 거래가 ○%", 나머지는 "1 늘면 ○%", 범주는 기준 범주(화곡동·철근콘크리트) 대비.
p값이 0.05보다 작으면 우연으로 보기 어려운 효과. 결정계수 R² = {ols.rsquared:.3f}.</p>
{table(coef)}
<div class="callout"><b>계수를 읽을 때 주의</b><ul>
<li>계수는 <b>다른 칸을 모두 고정했을 때</b>의 효과임. 공시가격(pub_ppm, 계수 {ols.params['pub_ppm']:.2f})이 이미 건물 나이·층 같은 개별요인을 반영하고 있어서
(건물 나이 ↔ 공시가격 상관 −0.71), 건물 나이 {coef.set_index('변수').loc['building_age', '해석']} · 층 {coef.set_index('변수').loc['floor', '해석']}처럼
EDA에서 본 방향과 반대로 나올 수 있음 → "공시가격이 같다면" 붙는 보정으로 읽어야 함. basis 문장에 그대로 쓰면 오해 소지 있음</li>
<li>pub_ppm_missing의 큰 +계수: 공시가격이 없던 행(주로 공시 뒤 신축)을 동 중앙값(낮은 값)으로 채웠기 때문에, 표시 칸이 그 차이를 보정하는 것</li>
<li>hhld_cnt_missing과 parking_per_hh_missing은 완전히 같은 행이라 두 계수가 효과를 반씩 나눠 가짐 → 둘을 합쳐서 읽어야 함</li></ul></div>
<h3>{tree_name} 변수 중요도</h3>{imp_fig}
<p class="muted">중요도는 "그 칸으로 나눴을 때 오차가 얼마나 줄었나"의 합(gain)을 비율로 나타낸 것. 방향(+/−)은 알려 주지 않음.</p></section>

<section id="s7"><h2>7. OLS와 ElasticNet의 차이</h2>{table(pd.DataFrame([
    ["무엇을 줄이나", "예측 오차만 최소화", "예측 오차 + 계수 크기에 대한 벌점"],
    ["벌점", "없음", "릿지(L2: 계수를 고르게 작게) + 라쏘(L1: 쓸모없는 계수를 0으로)를 섞음"],
    ["겹치는 변수", "계수가 불안정해질 수 있음 (VIF 높은 공시가격·주변 시세)", "벌점 덕분에 계수가 안정됨"],
    ["스케일링", "불필요 (예측 같음)", "필요 (벌점이 계수 크기에 걸리므로 단위가 다르면 불공평)"],
    ["계수 해석", "그대로 % 효과로 읽음", "벌점 때문에 0 쪽으로 줄어 있어 효과가 작게 나옴"],
    ["R로 치면", "lm()", "glmnet (alpha = 라쏘 비율)"],
], columns=["", "OLS", "ElasticNet"]))}
<p>둘 다 "직선 관계"를 가정하는 같은 회귀 모델이고, 차이는 <b>벌점(규제)</b> 유무뿐. 데이터가 많고 변수가 적으면 두 결과가 비슷하게 나옴.</p></section>

<section id="s8"><h2>8. 한계와 다음 단계</h2><ul>
<li>모든 모델이 <b>기본 설정</b> — 트리 모델은 튜닝 여지가 큼 (7단계)</li>
<li>ElasticNet의 자동 설정 고르기는 학습 조각 안에서 지번 구분 없이 나눔 → 설정값 선택이 약간 낙관적일 수 있음</li>
<li>price_low·price_high 구간과 confidence는 아직 없음 (6단계 이후)</li>
<li>시험용 2,200건 평가는 9단계에서 한 번만</li></ul></section>
</div></body></html>"""
(REPORT / "model_report.html").write_text(html, encoding="utf-8")
print(res.round(2).to_string(index=False))
print(f"저장: reports/model_compare.csv, reports/model_report.html")
