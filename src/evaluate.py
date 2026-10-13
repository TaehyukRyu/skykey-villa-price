"""9단계 최종 평가 — 시험용 2,200건(학습에 안 쓴 지번 20%)을 predict.py로 한 번 돌린 결과 채점 (결정기록.md 17번)

기준일 = 각 거래일 (10/13 사용자 결정 A): 거래 당시 알 수 있던 정보로 그날 가격을 추정 → 교차검증과 같은 조건이라 직접 비교 가능
대표 숫자 (결과 보기 전 결정): 시험용 전체 MAPE + 최근 12개월 MAPE. 결과를 보고 모델·규칙을 고치지 않음 (버그만 고치고 기록)
입력: data/processed/test_input.csv → predict.py → reports/test_output.csv, 정답 answer_all.csv
확인: predict.py 추정값 = 전처리 파일의 시험용 행을 같은 모델로 직접 예측한 값 (예측 경로가 학습 경로와 같은지)
예시 산출 3건 (사용자 결정): 시험용 최근 12개월 거래에서 구별 1건 무작위 (난수 42)
출력: reports/test_report.html, reports/test_results.csv
실행: python predict.py --input data/processed/test_input.csv --output reports/test_output.csv
      python src/evaluate.py
"""
import json
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tune_models import PROC, RECENT, REPORT, ROOT, build, img, table

P = json.loads((REPORT / "uncertainty_params.json").read_text(encoding="utf-8"))
TIERS = P["등급"]["이름"]

# ── 데이터: 입력 + predict.py 출력 + 정답 ───────────────────────
inp = pd.read_csv(PROC / "test_input.csv", dtype=str, keep_default_na=False)
out = pd.read_csv(REPORT / "test_output.csv", dtype={"id": str, "basis": str, "status": str})
ans = pd.read_csv(PROC / "answer_all.csv", dtype={"id": str})
d = inp.merge(out, on="id").merge(ans, on="id")
assert len(d) == len(inp) == 2200, "행 수가 맞지 않음"
fail = d[d.status != "ok"]
d = d[d.status == "ok"].reset_index(drop=True).astype({"price_est": int, "price_low": int, "price_high": int})
y, est = d.price_won.to_numpy(float), d.price_est.to_numpy(float)
ape = np.abs(est - y) / y
inside = (y >= d.price_low.to_numpy()) & (y <= d.price_high.to_numpy())
recent = (d.deal_date >= RECENT).to_numpy()
d["gu"] = d.sigungu.str.split().str[1]
# 등급: 출력의 confidence는 소수 2자리로 반올림돼 경계 근처에서 어긋날 수 있음 → 구간 하한 비율(등급마다 고정값)로 정확히 복원
ratio = np.log(d.price_low / d.price_est).to_numpy()
d["tier"] = [TIERS[int(np.argmin(np.abs(np.array(P["구간"]["log_하한"]) - v)))] for v in ratio]
conf = d.confidence.to_numpy(float)

# ── 확인: predict.py 추정 = 같은 모델로 전처리 파일의 시험용 행을 직접 예측 ────────────
pre = pd.read_csv(PROC / "preprocessed_all.csv").merge(pd.read_csv(PROC / "split.csv"), on="id")
tr, te = pre[pre.set == "train"], pre[pre.set == "test"].astype({"id": str}).set_index("id").loc[d.id]
params = json.loads((REPORT / "best_params.json").read_text(encoding="utf-8"))
cat = build("CatBoost", params["CatBoost"]).fit(tr, tr.price_won)
direct = np.round(cat.predict(te), -4)
same = int((direct == est).sum())
print(f"확인: predict.py 추정 = 직접 예측 {same}/{len(d)}건 (최대 차이 {np.abs(direct - est).max():,.0f}원)")

# 기준선: 동별 ㎡당 중앙값 × 면적 (학습용으로 계산, 학습용에 없는 동은 구 중앙값) — 5단계와 같은 방식
ppm = tr.price_won / tr.area_m2
base_ppm = te.dong.map(ppm.groupby(tr.dong).median()).fillna(te.gu.map(ppm.groupby(tr.gu).median())).to_numpy()
base_ape = np.abs(base_ppm * te.area_m2.to_numpy() - y) / y


def m(mask, a=ape):
    """성적: 건수, MAPE, 중앙값 오차율, ±10% 안, 구간 적중, 평균 confidence"""
    return {"건수": int(mask.sum()), "MAPE(%)": a[mask].mean() * 100, "중앙값 오차율(%)": np.median(a[mask]) * 100,
            "±10% 안(%)": (a[mask] <= .1).mean() * 100, "구간 적중(%)": inside[mask].mean() * 100,
            "평균 confidence": conf[mask].mean()}


ALL = np.ones(len(d), bool)
summary = pd.DataFrame([{"묶음": "시험용 전체", **m(ALL)}, {"묶음": "최근 12개월", **m(recent)}]
                       + [{"묶음": f"{g}", **m((d.gu == g).to_numpy())} for g in ["강남구", "강서구", "관악구"]]
                       + [{"묶음": f"최근 12개월 · {g}", **m(recent & (d.gu == g).to_numpy())} for g in ["강남구", "강서구", "관악구"]])
tiers = pd.DataFrame([{"묶음": f"{w} · 등급 {t}", **m(mask & (d.tier == t).to_numpy())}
                      for w, mask in [("최근 12개월", recent), ("전체", ALL)] for t in TIERS[::-1]])

# 교차검증(학습용)과 비교 — 같은 지표를 oof_predictions.csv로 계산
oof = pd.read_csv(PROC / "oof_predictions.csv").merge(pre[["id", "deal_date"]], on="id")
o_ape = np.abs(oof.pred_final - oof.price_won).to_numpy() / oof.price_won.to_numpy()
o_rec = (oof.deal_date >= RECENT).to_numpy()
cmp = pd.DataFrame([
    {"구분": "교차검증 (학습용 8,793건, 지번 단위 5조각)", "MAPE(%)": o_ape.mean() * 100, "중앙값 오차율(%)": np.median(o_ape) * 100,
     "±10% 안(%)": (o_ape <= .1).mean() * 100, "최근 12개월 MAPE(%)": o_ape[o_rec].mean() * 100},
    {"구분": f"시험용 ({len(d):,}건, 학습에 안 쓴 지번)", "MAPE(%)": ape.mean() * 100, "중앙값 오차율(%)": np.median(ape) * 100,
     "±10% 안(%)": (ape <= .1).mean() * 100, "최근 12개월 MAPE(%)": ape[recent].mean() * 100},
    {"구분": "기준선: 동별 ㎡당 중앙값 × 면적 (시험용)", "MAPE(%)": base_ape.mean() * 100, "중앙값 오차율(%)": np.median(base_ape) * 100,
     "±10% 안(%)": (base_ape <= .1).mean() * 100, "최근 12개월 MAPE(%)": base_ape[recent].mean() * 100}])

# confidence 정직성 (최근 12개월 = 규칙을 정한 기준과 같은 시기): 5등분
q = pd.qcut(conf[recent], 5, labels=False, duplicates="drop")
cal = pd.DataFrame([{"confidence 5등분": k + 1, "건수": int((q == k).sum()), "평균 confidence": conf[recent][q == k].mean(),
                     "실제 ±10% 적중률": (ape[recent][q == k] <= .1).mean(), "MAPE(%)": ape[recent][q == k].mean() * 100}
                    for k in range(int(q.max()) + 1)])

# 오차가 컸던 사례
d = d.assign(ape=ape * 100, err=(est / y - 1) * 100, inside=inside)
top = d.nlargest(10, "ape")
cases = pd.DataFrame({"id": top.id, "구": top.gu, "동": top.dong, "지번": top.jibun, "층": top.floor, "면적(㎡)": top.area_m2,
                      "거래일": top.deal_date, "실제(억)": (top.price_won / 1e8).round(2), "추정(억)": (top.price_est / 1e8).round(2),
                      "오차(%)": top.err.round(1), "confidence": top.confidence, "등급": top.tier,
                      "구간 안": np.where(top.inside, "O", "X"), "basis": top.basis})
big = (d.ape >= d.ape.quantile(.99)).to_numpy()
feat = te.reset_index()
traits = pd.DataFrame({"항목": ["면적(㎡)", "건물 나이(년)", "주변 거래 수", "공시가격 없음 비율(%)", "confidence", "실제가 > 추정 비율(%)", "강남구 비율(%)"],
                       f"오차 상위 1% ({int(big.sum())}건)": [feat.area_m2[big].median(), feat.building_age[big].median(), feat.nb_cnt_500m[big].median(),
                                                         feat.pub_ppm_missing[big].mean() * 100, np.median(conf[big]), (y > est)[big].mean() * 100,
                                                         (d.gu == "강남구")[big].mean() * 100],
                       "시험용 전체": [feat.area_m2.median(), feat.building_age.median(), feat.nb_cnt_500m.median(), feat.pub_ppm_missing.mean() * 100,
                                  np.median(conf), (y > est).mean() * 100, (d.gu == "강남구").mean() * 100]})

# 예시 산출 3건: 최근 12개월에서 구별 1건 무작위 (결과 보기 전에 정한 방법)
rng = np.random.default_rng(42)
ex = d.loc[[rng.choice(d.index[recent & (d.gu == g).to_numpy()]) for g in ["강남구", "강서구", "관악구"]]]
examples = pd.DataFrame({"입력": [f"{r.sigungu} {r.dong} {r.jibun}, {r.floor}층, 호 {r.ho or '없음'}, {r.area_m2}㎡" for r in ex.itertuples()],
                         "실제 거래 (거래일)": [f"{r.price_won / 1e8:.2f}억 ({r.deal_date})" for r in ex.itertuples()],
                         "price_est": ex.price_est, "price_low~high": [f"{r.price_low:,} ~ {r.price_high:,}" for r in ex.itertuples()],
                         "confidence": ex.confidence, "오차(%)": ex.err.round(1), "basis": ex.basis})

# ── 저장·그림·리포트 ─────────────────────────────────────
pd.concat([cmp.assign(표="교차검증 비교"), summary.assign(표="묶음별"), tiers.assign(표="등급별")]).round(3).to_csv(
    REPORT / "test_results.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(5.2, 4.4))
ax.plot([0, 1], [0, 1], color="#bbb", ls="--", lw=1, label="완벽하게 정직한 선")
ax.plot(cal["평균 confidence"], cal["실제 ±10% 적중률"], marker="o", color="#8e6bbf", label="5등분 묶음 (최근 12개월)")
lim = (min(cal["평균 confidence"].min(), cal["실제 ±10% 적중률"].min()) - .05, max(cal["평균 confidence"].max(), cal["실제 ±10% 적중률"].max()) + .05)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("모델이 매긴 confidence (묶음 평균)")
ax.set_ylabel("실제로 ±10% 안에 든 비율")
ax.legend(fontsize=8, loc="upper left")
ax.set_title("시험용 confidence의 정직성")
cal_fig = img(fig)

css = (ROOT / "src" / "eda.py").read_text(encoding="utf-8").split("<style>")[1].split("</style>")[0].replace("{{", "{").replace("}}", "}")
s_all, s_rec = summary.iloc[0], summary.iloc[1]
html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>최종 평가</title><style>{css}</style></head><body><div class="wrap">
<header><h1>다세대 시세 모델 — 최종 평가 (9단계, 시험용)</h1>
<p>시험용 {len(inp):,}건 (학습에 쓰지 않은 지번 20%, EDA 전에 분리) · 기준일 = 각 거래일 · predict.py로 한 번 실행 · 생성 {date.today()} · <code>python src/evaluate.py</code></p></header>
<nav><a href="#s0">한눈에</a><a href="#s1">1. 교차검증 비교</a><a href="#s2">2. 묶음별</a><a href="#s3">3. 등급별</a><a href="#s4">4. 정직성</a><a href="#s5">5. 오차 큰 사례</a><a href="#s6">6. 예시 3건</a><a href="#s7">7. 방법·주의</a></nav>
<section id="s0"><h2>한눈에 보기</h2><ol>
<li><b>대표 숫자:</b> 시험용 전체 MAPE <b>{s_all['MAPE(%)']:.2f}%</b> · 최근 12개월 MAPE <b>{s_rec['MAPE(%)']:.2f}%</b> (교차검증 {cmp['MAPE(%)'][0]:.2f}% · {cmp['최근 12개월 MAPE(%)'][0]:.2f}%)</li>
<li>중앙값 오차율 {s_all['중앙값 오차율(%)']:.2f}% · ±10% 안 {s_all['±10% 안(%)']:.1f}% · 기준선 MAPE {cmp['MAPE(%)'][2]:.2f}%</li>
<li>구간 적중 (목표 80%): 최근 12개월 {s_rec['구간 적중(%)']:.1f}% · 전체 {s_all['구간 적중(%)']:.1f}%</li>
<li>confidence: 최근 12개월 평균 {s_rec['평균 confidence']:.3f} vs 실제 ±10% 적중률 {s_rec['±10% 안(%)'] / 100:.3f}</li>
<li>산출 실패 {len(fail)}건 · predict.py 추정 = 직접 예측 {same:,}/{len(d):,}건</li></ol></section>
<section id="s1"><h2>1. 교차검증(학습용)과 비교 — 과대적합 확인</h2><p class="why">학습에 안 쓴 지번에서도 교차검증과 비슷하면, 처음 보는 물건에 일관되게 동작한다는 뜻</p>{table(cmp.round(2))}</section>
<section id="s2"><h2>2. 묶음별 성적</h2>{table(summary.round(2))}<p class="muted">구간 적중 = 실제 거래가가 price_low~price_high 안에 든 비율. 구간·신뢰도 규칙은 최근 12개월 거래로 정했으므로 그 이전 거래에선 보수적(넓게)일 수 있음</p></section>
<section id="s3"><h2>3. 신뢰도 등급별</h2>{table(tiers.round(2))}</section>
<section id="s4"><h2>4. confidence의 정직성 (최근 12개월)</h2><div style="display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start"><div>{cal_fig}</div><div>{table(cal.round(3))}</div></div></section>
<section id="s5"><h2>5. 오차가 컸던 사례</h2><h3>오차 상위 10건</h3>{table(cases)}<h3>오차 상위 1%와 전체 비교 (중앙값·비율)</h3>{table(traits.round(2))}
<p class="muted">사례별 원인은 데이터에 없는 정보(내부 상태, 거래 사정 등)일 수 있어 단정하지 않음</p></section>
<section id="s6"><h2>6. 예시 산출 3건 (최근 12개월에서 구별 1건 무작위, 난수 42)</h2>{table(examples)}</section>
<section id="s7"><h2>7. 방법·주의</h2><ul>
<li>시험용 = 3단계(실제로는 EDA 전)에 지번 단위로 20% 떼어 둔 거래. 같은 건물이 학습·시험에 동시에 없음(겹치는 지번 0개 확인)</li>
<li>기준일 = 각 거래일 (사용자 결정): 거래 당시 공개된 공시가격 연도·직전 12개월 주변 거래만 사용 → 교차검증과 같은 조건</li>
<li>회사 블라인드 검증은 기준일이 과제 안내일(2026-10-06)이라, 거래일과 안내일 사이 시장 변화만큼 오차가 더 생길 수 있음</li>
<li>결과를 본 뒤 모델·규칙을 고치지 않음. 시험용 이상치는 걸러내지 않음(실제 블라인드 검증처럼)</li></ul></section>
</div></body></html>"""
(REPORT / "test_report.html").write_text(html, encoding="utf-8")
print(cmp.round(2).to_string(index=False))
print(summary.round(2).to_string(index=False))
print(tiers.round(2).to_string(index=False))
print(cal.round(3).to_string(index=False))
print(f"산출 실패 {len(fail)}건" + (f": {fail.basis.value_counts().to_dict()}" if len(fail) else ""))
print("저장: reports/test_report.html, reports/test_results.csv")
