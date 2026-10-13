"""최종 모델 선택 — 튜닝된 트리 3개와 평균 조합 비교, 학습용만 (결정기록.md 13번)

후보 7개 (10/13 사용자 결정): CatBoost·XGBoost·LightGBM 단독 3 + 2개 평균 3 + 3개 평균 1
평균 방법 (사용자 결정): 모델마다 log(가격) 예측을 평균한 뒤 exp로 되돌림 — 모델이 log로 학습했으므로 그 상태에서 평균
가중 평균·스태킹 제외: 튜닝에 쓴 같은 조각으로 가중치까지 맞추면 점수가 더 부풀려짐. 단순 평균은 새로 맞추는 값이 없음
선택 규칙 (결과 보기 전 사용자 결정): 평균 모델이 CatBoost 단독보다 MAPE 0.1%p 이상 낮고 5조각 중 4조각 이상에서 이기면 평균 모델,
  아니면 CatBoost 단독. 여러 개가 통과하면 최저 MAPE에서 0.1%p 안에 드는 것 중 모델 수가 적은 것 (AI가 정한 세부 규칙, 단순한 쪽 우선)
조각·모델 만드는 함수는 tune_models.py 것을 그대로 씀 → 단독 모델 MAPE가 tuning_results.csv와 같아야 함 (실행 중 확인)
출력: data/processed/oof_predictions.csv (6단계 구간·confidence 재료), reports/select_report.html
실행: python src/select_model.py
"""
import json
import time
from datetime import date
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tune_models import COLOR, FOLDS, PROC, REPORT, ROOT, cv_oof, df, img, metrics, recent, table, y

TREES = ["CatBoost", "XGBoost", "LightGBM"]
BASE = "CatBoost"  # 비교 기준 = 튜닝 후 1위
MARGIN, MIN_WINS = 0.1, 4  # 선택 규칙: MAPE 0.1%p 이상 낮고, 5조각 중 4조각 이상에서 이김
GUS = {"강남구": "강남구", "강서구": "화곡동(강서구)", "관악구": "관악구"}

fold = np.zeros(len(df), int)
for k, (_, va_i) in enumerate(FOLDS):
    fold[va_i] = k


def fold_mape(p):
    return np.array([metrics(y[fold == k], p[fold == k])["MAPE(%)"] for k in range(5)])


# ── 1. 단독 모델 교차검증 예측 (튜닝된 설정) ─────────────────────────
params = json.loads((REPORT / "best_params.json").read_text(encoding="utf-8"))
tuned = pd.read_csv(REPORT / "tuning_results.csv").query("구분 == '튜닝 후'").set_index("모델")["MAPE(%)"]
logp = {}
for n in TREES:
    t0 = time.time()
    logp[n] = np.log(cv_oof(n, params[n]))
    got = metrics(y, np.exp(logp[n]))["MAPE(%)"]
    print(f"{n}: MAPE {got:.3f}% (튜닝 리포트 {tuned[n]:.3f}%) | {time.time() - t0:.0f}초", flush=True)
    assert abs(got - tuned[n]) < 0.001, f"{n} MAPE가 튜닝 결과와 다름 → 조각이나 설정이 바뀌었는지 확인"

# ── 2. 후보 7개 = 단독 3 + 평균 조합 4 (log에서 평균 → exp) ───────────
cands = {"+".join(c): np.exp(np.mean([logp[n] for n in c], axis=0)) for r in (1, 2, 3) for c in combinations(TREES, r)}

# ── 3. 비교표 ─────────────────────────────────────────
base_f, base_m = fold_mape(cands[BASE]), metrics(y, cands[BASE])["MAPE(%)"]
rows, frows, grows = [], [], []
for name, p in cands.items():
    m, fm, is_base = metrics(y, p), fold_mape(p), name == BASE
    rows.append({"후보": name, "모델 수": name.count("+") + 1, **m, "최근12개월 MAPE(%)": metrics(y[recent], p[recent])["MAPE(%)"],
                 "조각별 MAPE 표준편차": np.std(fm), f"{BASE} 대비(%p)": np.nan if is_base else m["MAPE(%)"] - base_m,
                 f"{BASE}를 이긴 조각": np.nan if is_base else int((fm < base_f).sum())})
    frows.append({"후보": name, **{f"조각 {k + 1}": v for k, v in enumerate(fm)}})
    grows.append({"후보": name, **{f"{label} MAPE(%)": metrics(y[(df.gu == g).to_numpy()], p[(df.gu == g).to_numpy()])["MAPE(%)"]
                                 for g, label in GUS.items()}})
res = pd.DataFrame(rows).sort_values("MAPE(%)").reset_index(drop=True)
order = list(res.후보)
folds = pd.DataFrame(frows).set_index("후보").loc[order].reset_index()
gus = pd.DataFrame(grows).set_index("후보").loc[order].reset_index()

# ── 4. 선택 규칙 적용 ──────────────────────────────────
avg = res[res["모델 수"] >= 2].copy()
c1, c2 = avg[f"{BASE} 대비(%p)"] <= -MARGIN, avg[f"{BASE}를 이긴 조각"] >= MIN_WINS
ok = avg[c1 & c2]
if len(ok):
    near = ok[ok["MAPE(%)"] <= ok["MAPE(%)"].min() + MARGIN]
    chosen = near.sort_values(["모델 수", "MAPE(%)"]).iloc[0]["후보"]
else:
    chosen = BASE
rule = avg[["후보", "MAPE(%)", f"{BASE} 대비(%p)", f"{BASE}를 이긴 조각"]].assign(  # True/False만 글자로 (replace는 숫자 1·0도 바꿈)
    **{"① MAPE 0.1%p 이상 낮음": np.where(c1, "통과", "—"), "② 4조각 이상 승리": np.where(c2, "통과", "—")})
pick = res.set_index("후보").loc[chosen]
WINS = f"{BASE}를 이긴 조각"


def show(d):
    """표시용: 소수 2자리, 이긴 조각은 '1/5'처럼 (기준 모델 자신은 —)"""
    return d.round(2).assign(**{WINS: d[WINS].map(lambda v: "—" if pd.isna(v) else f"{int(v)}/5")})

# ── 5. 모델끼리 엇갈림 vs 실제 오차 (6단계 confidence 미리보기) ──────────
L = np.column_stack([logp[n] for n in TREES])
spread = (np.exp(L.max(1) - L.min(1)) - 1) * 100  # 가장 높게 본 모델이 가장 낮게 본 모델보다 몇 % 높은가
ape = pd.Series(np.abs(cands[chosen] - y) / y)
bins = pd.qcut(spread, 5, labels=["1 (가장 일치)", "2", "3", "4", "5 (가장 엇갈림)"])
dis = pd.DataFrame({"구간": bins, "spread": spread, "ape": ape}).groupby("구간", observed=True).agg(
    건수=("ape", "size"), 엇갈림_최소=("spread", "min"), 엇갈림_최대=("spread", "max"),
    MAPE=("ape", lambda a: a.mean() * 100), 중앙값오차율=("ape", lambda a: a.median() * 100),
    안10=("ape", lambda a: (a <= .1).mean() * 100)).reset_index()
dis.columns = ["엇갈림 구간 (5등분)", "건수", "엇갈림 최소(%)", "엇갈림 최대(%)", "MAPE(%)", "중앙값 오차율(%)", "±10% 안 비율(%)"]
rho = pd.Series(spread).corr(ape, method="spearman")

# ── 6. 저장 ───────────────────────────────────────────
pd.DataFrame({"id": df.id, "fold": fold + 1, "price_won": y, **{f"pred_{n}": np.exp(logp[n]) for n in TREES},
              "pred_final": cands[chosen]}).to_csv(PROC / "oof_predictions.csv", index=False, encoding="utf-8-sig")

# ── 7. 그림 ───────────────────────────────────────────
def color(n):
    return COLOR.get(n, "#999")


fig, ax = plt.subplots(figsize=(10, 3.8))
x = np.arange(len(res))
ax.bar(x, res["MAPE(%)"], color=[color(n) for n in res.후보], edgecolor=["#000" if n == chosen else "none" for n in res.후보], lw=2)
for i, v in enumerate(res["MAPE(%)"]):
    ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
ax.axhline(base_m, color="#333", ls="--", lw=1, label=f"{BASE} 단독 {base_m:.2f}%")
ax.axhline(base_m - MARGIN, color="#d1495b", ls=":", lw=1, label=f"규칙 ① 기준선 {base_m - MARGIN:.2f}%")
ax.set_xticks(x, [n.replace("+", "+\n") for n in res.후보], fontsize=8)
ax.set_ylim(res["MAPE(%)"].min() - .5, res["MAPE(%)"].max() + .3)
ax.set_ylabel("MAPE(%)")
ax.legend(fontsize=8)
ax.set_title("후보별 MAPE (낮을수록 좋음, 검은 테두리 = 선택)")
bar_fig = img(fig)

fig, ax = plt.subplots(figsize=(10, 3.4))
avg_names = list(avg.후보)
for i, n in enumerate(avg_names):
    d = fold_mape(cands[n]) - base_f
    ax.scatter(np.full(5, i), d, color=["#2a9d8f" if v < 0 else "#d1495b" for v in d], s=30, zorder=3)
    ax.plot([i - .25, i + .25], [d.mean()] * 2, color="#333", lw=1.5)
ax.axhline(0, color="#999", lw=1)
ax.set_xticks(range(len(avg_names)), avg_names, fontsize=8)
ax.set_ylabel(f"{BASE} 대비 조각별 MAPE 차이(%p)")
ax.set_title(f"평균 모델의 조각별 성적 (점 = 조각, 0보다 아래 초록 = {BASE}보다 좋음, 가로선 = 평균)")
fold_fig = img(fig)

# ── 8. 리포트 ─────────────────────────────────────────
css = (ROOT / "src" / "eda.py").read_text(encoding="utf-8").split("<style>")[1].split("</style>")[0].replace("{{", "{").replace("}}", "}")
verdict = (f"평균 모델 <b>{chosen}</b> 선택 — 규칙 ①·② 모두 통과" if chosen != BASE
           else f"<b>{BASE} 단독</b> 유지 — 규칙 ①·②를 모두 통과한 평균 모델 없음")
html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>최종 모델 선택</title><style>{css}</style></head><body><div class="wrap">
<header><h1>다세대 시세 모델 — 최종 모델 선택</h1>
<p>학습용 {len(df):,}건 · 지번 단위 5조각 교차검증 (5·7단계와 같은 조각) · 튜닝된 설정(best_params.json) · 생성 {date.today()} · <code>python src/select_model.py</code></p></header>
<nav><a href="#s0">한눈에</a><a href="#s1">1. 방법</a><a href="#s2">2. 후보 비교</a><a href="#s3">3. 규칙 적용</a><a href="#s4">4. 조각별</a><a href="#s5">5. 구별</a><a href="#s6">6. 엇갈림과 오차</a><a href="#s7">7. 주의</a></nav>
<section id="s0"><h2>한눈에 보기</h2><ol>
<li>{verdict}</li>
<li>선택 모델: MAPE {pick['MAPE(%)']:.2f}% · 중앙값 오차율 {pick['중앙값 오차율(%)']:.2f}% · ±10% 안 {pick['±10% 안 비율(%)']:.1f}% · 최근 12개월 MAPE {pick['최근12개월 MAPE(%)']:.2f}%</li>
<li>세 모델 예측이 엇갈리는 거래일수록 실제 오차가 큰가: 순위 상관 {rho:.2f} · 가장 일치하는 20% MAPE {dis['MAPE(%)'].iloc[0]:.1f}% vs 가장 엇갈리는 20% {dis['MAPE(%)'].iloc[-1]:.1f}%</li></ol></section>
<section id="s1"><h2>1. 방법</h2>{table(pd.DataFrame([
    ["후보 7개 (사용자 결정)", "CatBoost·XGBoost·LightGBM 단독 3개 + 2개씩 평균 3개 + 3개 평균 1개. 단독 모델은 튜닝된 설정으로 5조각 학습, 조합은 그 예측을 평균만 냄(새 학습 없음)"],
    ["평균 방법 (사용자 결정)", "모델마다 log(가격) 예측을 평균한 뒤 원 단위로 되돌림. 모델이 log로 학습했으므로 그 상태에서 평균 → 나중에 요인별 기여도 분해가 평균에도 그대로 더해짐"],
    ["선택 규칙 (결과 보기 전 사용자 결정)", f"평균 모델이 {BASE} 단독보다 ① MAPE {MARGIN}%p 이상 낮고 ② 5조각 중 {MIN_WINS}조각 이상에서 이기면 평균 모델, 아니면 {BASE} 단독 (단순한 쪽 우선)"],
    ["여러 개가 통과할 때 (AI가 정한 세부 규칙)", f"가장 낮은 MAPE에서 {MARGIN}%p 안에 드는 것 중 모델 수가 적은 것"],
    ["제외한 방법", "가중 평균·스태킹(모델 위에 모델을 얹어 가중치 학습): 튜닝에 쓴 같은 조각으로 가중치까지 맞추면 점수가 실제보다 좋게 나옴"],
    ["확인", "단독 모델 3개의 MAPE가 튜닝 리포트(tuning_results.csv)와 소수 셋째 자리까지 같음 (실행 중 자동 확인)"],
], columns=["항목", "내용"]))}</section>
<section id="s2"><h2>2. 후보 비교 (MAPE 낮은 순)</h2>{bar_fig}{table(show(res))}
<p class="muted">{BASE} 대비 = 음수면 {BASE} 단독보다 좋음. 이긴 조각 = 5개 검증 조각 중 {BASE}보다 MAPE가 낮았던 조각 수. 최근 12개월 = {int(recent.sum()):,}건.</p></section>
<section id="s3"><h2>3. 선택 규칙 적용 (평균 모델 4개)</h2>{table(show(rule))}<p><b>결과:</b> {verdict}</p></section>
<section id="s4"><h2>4. 조각별 MAPE</h2>{fold_fig}{table(folds.round(2))}</section>
<section id="s5"><h2>5. 구별 MAPE</h2>{table(gus.round(2))}</section>
<section id="s6"><h2>6. 모델끼리 엇갈림 vs 실제 오차 (6단계 confidence 미리보기)</h2>
<p class="why">엇갈림 = 세 모델 중 가장 높게 본 예측이 가장 낮게 본 예측보다 몇 % 높은가. 오차는 선택 모델({chosen}) 기준. 엇갈림이 큰 구간에서 오차도 커지면, 예측할 때 이 값으로 신뢰도를 낮출 수 있음 (정답 없이도 계산 가능).</p>
{table(dis.round(2))}<p>순위 상관(스피어만) {rho:.3f}</p></section>
<section id="s7"><h2>7. 주의</h2><ul>
<li>튜닝과 이 비교에 같은 5조각을 씀 → 모든 후보 점수가 약간 낙관적. 후보끼리 비교는 같은 조건이라 공정하지만, 최종 성능은 9단계 시험용 2,200건으로 한 번만 측정</li>
<li>후보를 7개 비교하면 우연히 좋아 보이는 후보를 고를 위험이 생김 → 0.1%p 차이 + 4조각 이상 승리 규칙을 결과 보기 전에 정해 둠</li>
<li>교차검증 예측은 data/processed/oof_predictions.csv에 저장 (pred_final = 선택 모델 {chosen}) → 6단계 구간(price_low·price_high)과 confidence 계산 재료</li></ul></section>
</div></body></html>"""
(REPORT / "select_report.html").write_text(html, encoding="utf-8")
print(show(res).to_string(index=False))
print(show(rule).to_string(index=False))
print(dis.round(2).to_string(index=False))
print(f"선택: {chosen} | 엇갈림-오차 순위 상관 {rho:.3f}")
print("저장: data/processed/oof_predictions.csv, reports/select_report.html")
