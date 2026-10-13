"""6단계 구간·신뢰도 — price_low·price_high와 confidence 계산 규칙, 학습용 교차검증 예측으로만 (결정기록.md 14번)

입력: data/processed/oof_predictions.csv (각 거래를 그 거래를 안 본 CatBoost가 맞힌 값, select_model.py)
규칙 기준 (사용자 결정): 최근 12개월 거래만 — 블라인드 검증 물건과 같은 시기. 3년 전체로 정하면 최근 거래에서 구간 적중 76.6%
confidence (10/13 사용자 결정): 실제가가 추정가 ±10% 안에 들 확률. 로지스틱 회귀 = R의 glm(적중 ~ 재료, family=binomial)
  재료 (사용자 결정): log(1+주변 거래 수), 세 모델 엇갈림 %(그대로), 권역(기준 강서구 = 학습용 최다, AI가 정함)
  엇갈림 = CatBoost·XGBoost·LightGBM 예측 중 최고가 최저보다 몇 % 높은가 → predict.py도 세 모델을 돌려야 함
구간 (사용자 결정): 오차 분포 + 신뢰도 등급별. 등급마다 과거 오차(실제 ÷ 추정)의 하위 10%·상위 10% 지점 → 가운데 80%
  등급 = confidence 3등분 (낮음·보통·높음 — 등급마다 약 1,200건이 되도록 AI가 정한 세부 규칙)
평가: 이중 교차검증 — 4조각의 최근 거래로 confidence·구간을 정하고 나머지 1조각에서 채점, 5번 반복 (자기 답안지로 채점 방지)
출력: reports/uncertainty_params.json (학습용 최근 12개월로 정한 규칙, predict.py가 읽음), reports/uncertainty_report.html
실행: python src/uncertainty.py
"""
import json
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

from tune_models import PROC, RECENT, REPORT, ROOT, img, table

HIT, COVER = 0.10, 0.80  # confidence = 오차율 10% 이하일 확률 / 구간 목표 = 80%
TIERS = ["낮음", "보통", "높음"]
GUS = ["강남구", "관악구"]  # 권역 더미 (기준 = 강서구)
MODELS = ["CatBoost", "XGBoost", "LightGBM"]
COLS = ["상수", "log(1+주변 거래 수)", "엇갈림(%)"] + [f"권역: {g}" for g in GUS]


def spread_pct(preds):
    """세 모델 엇갈림: 가장 높은 예측이 가장 낮은 예측보다 몇 % 높은가 (select_model.py와 같은 계산)"""
    L = np.log(np.column_stack(preds))
    return (np.exp(L.max(1) - L.min(1)) - 1) * 100


def design(nb_cnt, spread, gu):
    """로지스틱 회귀 입력 (predict.py도 같은 순서): 상수, log(1+주변 거래 수), 엇갈림(%), 강남구, 관악구"""
    gu = np.asarray(gu)
    return np.column_stack([np.ones(len(gu)), np.log1p(nb_cnt), spread] + [(gu == g).astype(float) for g in GUS])


def fit(rows):
    """rows로 규칙을 정함: confidence 계수, 등급 경계(confidence 3등분), 등급별 오차 하위·상위 10% 지점(log)"""
    lr = sm.Logit(hit[rows], X[rows]).fit(disp=0)
    conf = lr.predict(X[rows])
    cuts = np.quantile(conf, [1 / 3, 2 / 3])
    tier = np.digitize(conf, cuts)
    q = np.array([np.quantile(r[rows][tier == t], [(1 - COVER) / 2, (1 + COVER) / 2]) for t in range(3)])
    return lr, cuts, q


def apply(coef, cuts, q, Xa, est_a):
    """규칙 적용 → confidence, 등급(0 낮음·1 보통·2 높음), price_low, price_high"""
    conf = 1 / (1 + np.exp(-Xa @ coef))
    tier = np.digitize(conf, cuts)
    return conf, tier, est_a * np.exp(q[tier, 0]), est_a * np.exp(q[tier, 1])


def summary(m):
    """행 묶음 m의 성적: 건수, 실제 ±10% 적중률, MAPE, 구간 안 비율, 평균 하한·상한(추정 대비 %)"""
    return {"건수": int(m.sum()), "평균 confidence": conf[m].mean(), "실제 ±10% 적중률(%)": hit[m].mean() * 100,
            "MAPE(%)": ape[m].mean() * 100, "구간 안 비율(%)": inside[m].mean() * 100,
            "평균 하한(%)": (low[m] / est[m] - 1).mean() * 100, "평균 상한(%)": (high[m] / est[m] - 1).mean() * 100}


# ── 데이터 ────────────────────────────────────────────
o = pd.read_csv(PROC / "oof_predictions.csv")
p = pd.read_csv(PROC / "preprocessed_all.csv").drop(columns="price_won")
jb = pd.read_csv(PROC / "input_all.csv", dtype=str)[["id", "jibun"]].astype({"id": int})
d = o.merge(p, on="id").merge(jb, on="id")
assert len(d) == len(o), "id 매칭 실패"
y, est = d.price_won.to_numpy(), d.pred_final.to_numpy()
ape = np.abs(est - y) / y
hit = (ape <= HIT).astype(int)
r = np.log(y / est)  # 오차(log): 실제가 추정의 몇 배였나
recent = (d.deal_date >= RECENT).to_numpy()
spread = spread_pct([d[f"pred_{m}"] for m in MODELS])
X = design(d.nb_cnt_500m.to_numpy(), spread, d.gu)

# ── 이중 교차검증: 다른 4조각의 최근 12개월 거래로 규칙을 정하고 나머지 1조각에 적용 ─────────────
conf, tier, low, high = np.zeros(len(d)), np.zeros(len(d), int), np.zeros(len(d)), np.zeros(len(d))
one_low, one_high = np.zeros(len(d)), np.zeros(len(d))  # 참고: 등급 없이 구간 하나
for k in sorted(d.fold.unique()):
    te = (d.fold == k).to_numpy()
    base = ~te & recent  # 규칙 기준 = 최근 12개월 거래 (블라인드 검증 물건과 같은 시기, 10/13 사용자 결정)
    lr_k, cuts_k, q_k = fit(base)
    conf[te], tier[te], low[te], high[te] = apply(lr_k.params, cuts_k, q_k, X[te], est[te])
    q1 = np.quantile(r[base], [(1 - COVER) / 2, (1 + COVER) / 2])
    one_low[te], one_high[te] = est[te] * np.exp(q1[0]), est[te] * np.exp(q1[1])
inside = (y >= low) & (y <= high)
one_inside = (y >= one_low) & (y <= one_high)

# ── 최종 규칙 (학습용의 최근 12개월 거래 전체) → JSON, 파일만으로 같은 값이 나오는지 확인 ──────────
lr, cuts, q = fit(recent)
params = {
    "설명": "6단계 구간·신뢰도 규칙 (src/uncertainty.py, 결정기록.md 14번). 학습용 최근 12개월 거래의 교차검증 예측으로 정함",
    "confidence": {"뜻": f"실제가가 추정가 ±{HIT:.0%} 안에 들 확률", "입력": COLS, "계수": lr.params.tolist(),
                   "식": "1 / (1 + exp(-(계수 · [1, log(1+nb_cnt_500m), 엇갈림%, 강남구 여부, 관악구 여부])))",
                   "엇갈림%": "(CatBoost·XGBoost·LightGBM 예측 중 최고 ÷ 최저 − 1) × 100"},
    "등급": {"이름": TIERS, "경계": cuts.tolist(), "설명": "confidence < 경계[0] → 낮음, < 경계[1] → 보통, 나머지 → 높음"},
    "구간": {"목표": COVER, "log_하한": q[:, 0].tolist(), "log_상한": q[:, 1].tolist(),
             "식": "price_low = 추정 × exp(log_하한[등급]), price_high = 추정 × exp(log_상한[등급])"},
    "기준": f"{RECENT} 이후 거래 (블라인드 검증 물건과 같은 시기)", "학습 건수": int(recent.sum()), "생성": str(date.today()),
}
(REPORT / "uncertainty_params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
P = json.loads((REPORT / "uncertainty_params.json").read_text(encoding="utf-8"))
c_mem, t_mem, l_mem, h_mem = apply(lr.params, cuts, q, X, est)
c_js, t_js, l_js, h_js = apply(np.array(P["confidence"]["계수"]), np.array(P["등급"]["경계"]),
                               np.column_stack([P["구간"]["log_하한"], P["구간"]["log_상한"]]), X, est)
assert np.allclose(c_js, lr.predict(X)) and np.allclose(c_js, c_mem) and (t_js == t_mem).all() and np.allclose(l_js, l_mem) and np.allclose(h_js, h_mem)
print(f"확인: JSON만으로 계산한 confidence·등급·구간 = 메모리 값 ({len(d):,}건)")

# ── 평가표 ────────────────────────────────────────────
# 채점 기준 = 최근 12개월 거래 (규칙을 정한 기준과 같음). 그 이전 거래는 참고로만
TN, R = np.array(TIERS), recent
tier_tab = pd.DataFrame([{"등급": TIERS[t], "confidence 범위": f"{conf[R & (tier == t)].min():.2f}~{conf[R & (tier == t)].max():.2f}",
                          **summary(R & (tier == t))} for t in (2, 1, 0)])
cover_tab = pd.DataFrame([{"묶음": "최근 12개월 (채점 기준)", **summary(R)}]
                         + [{"묶음": f"최근 12개월 · {g}", **summary(R & (d.gu == g).to_numpy())} for g in ["강남구", "강서구", "관악구"]]
                         + [{"묶음": "참고: 그 이전 거래", **summary(~R)}])
one = {"건수": int(R.sum()), "구간 안 비율(%)": one_inside[R].mean() * 100, "평균 하한(%)": (one_low[R] / est[R] - 1).mean() * 100,
       "평균 상한(%)": (one_high[R] / est[R] - 1).mean() * 100}
cr, hr, ar = conf[R], hit[R], ape[R]
dec = pd.qcut(cr, 10, labels=False)
cal = pd.DataFrame([{"confidence 10등분": k + 1, "건수": int((dec == k).sum()), "평균 confidence": cr[dec == k].mean(),
                     "실제 ±10% 적중률": hr[dec == k].mean(), "MAPE(%)": ar[dec == k].mean() * 100} for k in range(10)])
auc = roc_auc_score(hr, cr)
rho = pd.Series(cr).corr(pd.Series(ar), method="spearman")

coef = pd.DataFrame({"재료": COLS, "계수": lr.params, "표준오차": lr.bse, "p값": lr.pvalues, "오즈비 exp(계수)": np.exp(lr.params)})
coef["쉬운 뜻"] = ["—", "주변 거래가 약 2.7배 많아질 때 적중 오즈가 몇 배가 되나", "세 모델 차이가 1%p 커질 때 적중 오즈가 몇 배가 되나",
                 "강서구보다 적중 오즈가 몇 배인가", "강서구보다 적중 오즈가 몇 배인가"]
nb_r, sp_r = d.nb_cnt_500m[R], spread[R]
ex_in = [("좋은 조건", nb_r.quantile(.75), np.quantile(sp_r, .2), "강서구"),
         ("보통 조건", nb_r.median(), np.median(sp_r), "관악구"),
         ("나쁜 조건", nb_r.quantile(.05), np.quantile(sp_r, .8), "강남구")]
ex_c, ex_t, ex_l, ex_h = apply(lr.params, cuts, q, design([e[1] for e in ex_in], [e[2] for e in ex_in], [e[3] for e in ex_in]), np.ones(3))
examples = pd.DataFrame([{"예시": e[0], "주변 거래 수": int(e[1]), "엇갈림(%)": round(e[2], 1), "권역": e[3], "confidence": round(c, 2),
                          "등급": TIERS[t], "구간 (추정 대비)": f"{(lo - 1) * 100:+.0f}% ~ {(hi - 1) * 100:+.0f}%"}
                         for e, c, t, lo, hi in zip(ex_in, ex_c, ex_t, ex_l, ex_h)])

# 오차가 컸던 사례 (최근 12개월): 상위 10건 + 상위 1%와 전체 비교
d = d.assign(ape=ape * 100, conf=conf, tier=TN[tier], spread=spread, inside=inside)[R]
top = d.nlargest(10, "ape")
cases = pd.DataFrame({"id": top.id, "구": top.gu, "동": top.dong, "지번": top.jibun, "면적(㎡)": top.area_m2, "층": top.floor,
                      "건물 나이": top.building_age.round(1), "거래일": top.deal_date, "실제(억)": (top.price_won / 1e8).round(2),
                      "추정(억)": (top.pred_final / 1e8).round(2), "오차(%)": ((top.pred_final / top.price_won - 1) * 100).round(1),
                      "confidence": top.conf.round(2), "등급": top.tier, "주변 거래": top.nb_cnt_500m, "엇갈림(%)": top.spread.round(1),
                      "구간 안": np.where(top.inside, "O", "X")})
big = d.ape >= d.ape.quantile(.99)
traits = ["area_m2", "floor", "building_age", "nb_cnt_500m", "spread", "conf"]
names = ["면적(㎡)", "층", "건물 나이(년)", "주변 거래 수", "엇갈림(%)", "confidence"]
trait_tab = pd.DataFrame({"항목": names + ["실제가 > 추정 비율(%)", "강남구 비율(%)", "낮음 등급 비율(%)"],
                          f"오차 상위 1% ({int(big.sum())}건) 중앙값": [d.loc[big, c].median() for c in traits]
                          + [(d.price_won > d.pred_final)[big].mean() * 100, (d.gu == "강남구")[big].mean() * 100, (d.tier == "낮음")[big].mean() * 100],
                          "전체 중앙값": [d[c].median() for c in traits]
                          + [(d.price_won > d.pred_final).mean() * 100, (d.gu == "강남구").mean() * 100, (d.tier == "낮음").mean() * 100]})

# ── 그림: confidence가 정직한가 (10등분별 평균 confidence vs 실제 적중률) ──────────
fig, ax = plt.subplots(figsize=(5.2, 4.4))
ax.plot([0, 1], [0, 1], color="#bbb", ls="--", lw=1, label="완벽하게 정직한 선")
ax.plot(cal["평균 confidence"], cal["실제 ±10% 적중률"], marker="o", color="#8e6bbf", label="10등분 묶음")
lo_ax, hi_ax = min(cal["평균 confidence"].min(), cal["실제 ±10% 적중률"].min()) - .05, max(cal["평균 confidence"].max(), cal["실제 ±10% 적중률"].max()) + .05
ax.set_xlim(lo_ax, hi_ax)
ax.set_ylim(lo_ax, hi_ax)
ax.set_xlabel("모델이 매긴 confidence (묶음 평균)")
ax.set_ylabel("실제로 ±10% 안에 든 비율")
ax.legend(fontsize=8, loc="upper left")
ax.set_title("confidence의 정직성 (점이 점선 위에 있을수록 정직)")
cal_fig = img(fig)

# ── 리포트 ────────────────────────────────────────────
css = (ROOT / "src" / "eda.py").read_text(encoding="utf-8").split("<style>")[1].split("</style>")[0].replace("{{", "{").replace("}}", "}")
t_hi, t_lo = tier_tab.iloc[0], tier_tab.iloc[2]
html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>구간·신뢰도</title><style>{css}</style></head><body><div class="wrap">
<header><h1>다세대 시세 모델 — 구간(price_low·price_high)과 신뢰도(confidence) (6단계)</h1>
<p>학습용 교차검증 예측(CatBoost) · 규칙·채점 기준 = 최근 12개월 거래 {int(R.sum()):,}건 · 이중 교차검증으로 채점 · 생성 {date.today()} · <code>python src/uncertainty.py</code></p></header>
<nav><a href="#s0">한눈에</a><a href="#s1">1. 방법</a><a href="#s2">2. 정직성</a><a href="#s3">3. 등급별</a><a href="#s4">4. 구간 적중</a><a href="#s5">5. 계수·예시</a><a href="#s6">6. 오차 큰 사례</a><a href="#s7">7. 주의</a></nav>
<section id="s0"><h2>한눈에 보기</h2><ol>
<li><b>구간 적중률 {inside[R].mean() * 100:.1f}%</b> (최근 12개월, 목표 {COVER:.0%}) · 참고: 그 이전 거래 {inside[~R].mean() * 100:.1f}%</li>
<li><b>confidence 정직성:</b> 평균 confidence {cr.mean():.3f} vs 실제 ±10% 적중률 {hr.mean():.3f} · 구분력(AUC) {auc:.3f} (0.5 = 동전 던지기, 1 = 완벽)</li>
<li><b>등급 차이:</b> 높음 등급 MAPE {t_hi['MAPE(%)']:.1f}% · 구간 {t_hi['평균 하한(%)']:+.0f}~{t_hi['평균 상한(%)']:+.0f}% vs 낮음 등급 MAPE {t_lo['MAPE(%)']:.1f}% · 구간 {t_lo['평균 하한(%)']:+.0f}~{t_lo['평균 상한(%)']:+.0f}%</li>
<li>참고: 등급 없이 구간 하나로 하면 {one['평균 하한(%)']:+.0f}~{one['평균 상한(%)']:+.0f}%, 적중률 {one['구간 안 비율(%)']:.1f}%</li></ol></section>
<section id="s1"><h2>1. 방법</h2>{table(pd.DataFrame([
    ["confidence 뜻 (사용자 결정)", f"실제가가 추정가 ±{HIT:.0%} 안에 들 확률. 지금까지 모든 표의 '±10% 안 비율'과 같은 자 → 0.6이면 '이런 물건은 10건 중 6건이 ±10% 안'"],
    ["confidence 계산 (사용자 결정)", "로지스틱 회귀 = R의 glm(적중 ~ 재료, family=binomial). 적중 = 오차율 10% 이하면 1, 아니면 0"],
    ["재료 (사용자 결정)", "① log(1+주변 거래 수): 반경 500m·최근 12개월 다른 지번 거래 건수. 적을 땐 크게, 많아지면 평평한 모양이라 log ② 세 모델 엇갈림(%): 그대로 — 크게 엇갈릴 때만 급하게 나빠지는 모양 ③ 권역 (기준 강서구 = 학습용 최다, AI가 정함). 공시가격 찾은 수준은 적중률 차이가 거의 없어(호까지 50% vs 층까지 49%) 제외"],
    ["구간 (사용자 결정)", f"오차 분포 + 등급별: 등급마다 과거 오차(실제 ÷ 추정)의 하위 10%·상위 10% 지점 → 가운데 {COVER:.0%}가 들어가는 범위. 믿을 만한 물건은 좁게, 불확실한 물건은 넓게"],
    ["등급 (AI가 정한 세부 규칙)", "confidence 3등분 → 낮음·보통·높음 (등급마다 약 1,200건이라 구간 계산이 안정적)"],
    ["규칙 기준 (사용자 결정)", f"최근 12개월({RECENT} 이후) 거래 {int(R.sum()):,}건. 처음엔 3년 전체로 정했더니 최근 거래에서 구간 적중 76.6%·confidence 과신(평균 0.51 vs 실제 45.7%). 최근 거래는 치우침 없이(실제÷추정 중앙값 1.00) 위쪽으로 더 크게 흩어짐(상위 10% 지점 1.26~1.28배 vs 그 이전 1.20~1.24배) → 블라인드 검증 물건과 같은 시기로 맞춤 (연립 제외 때와 같은 원칙)"],
    ["채점 방법", "이중 교차검증: 5조각 중 4조각의 최근 거래로 confidence 계수·등급 경계·구간 폭을 정하고 나머지 1조각에서 채점, 5번 반복. 모든 표의 숫자는 이 방식 (규칙을 정할 때 본 거래는 채점에 안 씀)"],
    ["predict.py용 규칙", "학습용 최근 12개월 전체로 다시 정해 reports/uncertainty_params.json에 저장. 파일만으로 계산한 값이 메모리 값과 같은지 실행 중 확인"],
], columns=["항목", "내용"]))}</section>
<section id="s2"><h2>2. confidence가 정직한가</h2><p class="why">confidence를 10묶음으로 나눠, 모델이 매긴 값(평균)과 실제로 ±10% 안에 든 비율을 비교. 둘이 같으면 정직 — 과제 평가의 "신뢰도와 실제 오차의 일치"</p>
<div style="display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start"><div>{cal_fig}</div><div>{table(cal.round(3))}</div></div>
<p>순위 상관(confidence ↔ 오차율, 스피어만) {rho:.3f} — 음수면 confidence가 높을수록 오차가 작다는 뜻</p></section>
<section id="s3"><h2>3. 등급별 성적</h2>{table(tier_tab.round(2))}</section>
<section id="s4"><h2>4. 구간 적중률 (목표 {COVER:.0%})</h2>{table(cover_tab.round(2))}
<p class="muted">평균 하한·상한 = price_low·price_high가 추정가보다 평균 몇 % 낮고 높은가. 참고 — 등급 없이 하나로: 구간 {one['평균 하한(%)']:+.1f}~{one['평균 상한(%)']:+.1f}%, 적중률 {one['구간 안 비율(%)']:.1f}%</p></section>
<section id="s5"><h2>5. confidence 계수와 예시 (학습용 최근 12개월로 정한 최종 규칙)</h2>{table(coef.round(4))}
<p class="muted">오즈 = 맞을 확률 ÷ 틀릴 확률. 오즈비 1보다 크면 confidence를 올리고, 작으면 내림.</p>
<h3>예시 3건</h3><p class="why">좋은 조건 = 주변 거래 상위 25% 지점·엇갈림 하위 20% 지점·강서구 / 보통 = 둘 다 중앙값·관악구 / 나쁜 조건 = 주변 거래 하위 5% 지점·엇갈림 상위 20% 지점·강남구</p>{table(examples)}</section>
<section id="s6"><h2>6. 오차가 컸던 사례 (최근 12개월)</h2><h3>오차 상위 10건</h3>{table(cases)}
<h3>오차 상위 1%와 전체 비교</h3>{table(trait_tab.round(2))}
<p class="muted">사례별 원인은 데이터에 없는 정보(내부 상태, 거래 사정 등)일 수 있어 단정하지 않음. 위 표는 공통점만 보여줌.</p></section>
<section id="s7"><h2>7. 주의</h2><ul>
<li>규칙을 최근 12개월 거래로만 정해 3년 전체보다 계수의 흔들림(표준오차)이 큼 — 5번 표의 표준오차 참고</li>
<li>교차검증 예측은 학습용 4/5로 학습한 모델의 값 → predict.py의 최종 모델(학습 데이터가 더 많음)은 오차가 조금 작을 수 있어 구간이 약간 넓게(보수적으로) 잡힐 수 있음 (추정). 9단계 시험용으로 확인</li>
<li>엇갈림을 쓰므로 predict.py는 CatBoost 외에 XGBoost·LightGBM도 학습·실행해야 함 (추정값은 CatBoost만 사용)</li>
<li>면적을 입력 대신 건축물대장에서 찾았거나 못 찾은 경우처럼 학습 데이터에 없는 상황은 이 규칙이 반영하지 못함 → predict.py에서 따로 처리 필요</li></ul></section>
</div></body></html>"""
(REPORT / "uncertainty_report.html").write_text(html, encoding="utf-8")
print(tier_tab.round(2).to_string(index=False))
print(cover_tab.round(2).to_string(index=False))
print(cal.round(3).to_string(index=False))
print(coef.drop(columns="쉬운 뜻").round(4).to_string(index=False))
print(examples.to_string(index=False))
print(f"AUC {auc:.3f} | 순위 상관 {rho:.3f} | 등급 없이 하나로: {one['평균 하한(%)']:+.1f}~{one['평균 상한(%)']:+.1f}%, 적중 {one['구간 안 비율(%)']:.1f}%")
print("저장: reports/uncertainty_params.json, reports/uncertainty_report.html")
