"""README 그림 3개 — 결과 파일의 숫자를 그대로 읽어 그림 (10/13 사용자 요청)

입력: reports/tuning_results.csv·model_compare.csv (교차검증), reports/test_output.csv·test_results.csv (시험용),
      data/processed/answer_all.csv (시험용 실제 거래가)
출력: docs/img/models.png, actual_vs_pred.png, region.png
그림 하나에는 같은 기준의 숫자만 넣음 (교차검증끼리, 시험용끼리)
실행: python src/readme_figures.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORT, OUT = ROOT / "reports", ROOT / "docs" / "img"
INK, ACCENT, GRAY = "#1B2430", "#A8432F", "#C4BFB3"  # PPT와 같은 색
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"저장: docs/img/{name}")


def models():
    """7개 모델 평균 오차율 — 모두 학습용 8,793건 지번 단위 5조각 교차검증. 튜닝한 5개는 튜닝 후 값"""
    t = pd.read_csv(REPORT / "tuning_results.csv")
    t = t[t.구분 == "튜닝 후"].set_index("모델")["MAPE(%)"]
    c = pd.read_csv(REPORT / "model_compare.csv").set_index("모델")["MAPE(%)"]
    s = pd.concat([t, c[["OLS", "기준선"]]]).rename({"기준선": "단순 계산"}).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(s.index, s.values, color=[ACCENT if n == "CatBoost" else GRAY for n in s.index], height=0.6)
    for y, v in enumerate(s.values):
        ax.text(v + 0.4, y, f"{v:.1f}%", va="center")
    ax.set_xlim(0, s.max() * 1.15)
    ax.set_xlabel("평균 오차율 (MAPE, 낮을수록 좋음)")
    ax.set_title("7개 모델 비교 (학습 데이터 8,793건을 지번 단위 5조각으로 번갈아 시험)", loc="left", fontsize=11)
    ax.text(0, -0.28, "단순 계산 = 같은 동 거래의 ㎡당 가격 중앙값 × 면적", transform=ax.transAxes, fontsize=9, color="#4A5260")
    save(fig, "models.png")


def actual_vs_pred():
    """시험용 2,200건 실제 vs 추정 (9단계 평가: 학습용만으로 학습한 모델, 기준일 = 거래일)"""
    d = pd.read_csv(REPORT / "test_output.csv").merge(pd.read_csv(ROOT / "data" / "processed" / "answer_all.csv"), on="id")
    a, e = d.price_won / 1e8, d.price_est / 1e8
    inside = ((e / a - 1).abs() <= 0.1).mean()
    lim = [min(a.min(), e.min()) * 0.8, max(a.max(), e.max()) * 1.2]
    x = np.geomspace(*lim, 50)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.fill_between(x, x * 0.9, x * 1.1, color=ACCENT, alpha=0.18, lw=0, label="추정이 실제의 ±10% 안")
    ax.plot(x, x, color=ACCENT, lw=1)
    ax.scatter(a, e, s=6, alpha=0.35, color=INK, lw=0)
    ticks = [t for t in [0.3, 0.5, 1, 2, 3, 5, 10, 20] if lim[0] <= t <= lim[1]]
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="실제 거래가 (억원)", ylabel="추정 시세 (억원)")
    ax.minorticks_off()
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
    ax.set_title(f"시험용 {len(d):,}건: 실제 vs 추정 (±10% 안 {inside:.1%})", loc="left", fontsize=11)
    ax.legend(loc="upper left", frameon=False)
    save(fig, "actual_vs_pred.png")


def region():
    """권역별 평균 오차율 (시험용) + 시험용 전체"""
    r = pd.read_csv(REPORT / "test_results.csv")
    r = r[r.표 == "묶음별"].set_index("묶음")
    names = {"강서구": "강서구 화곡동", "관악구": "관악구", "강남구": "강남구"}
    v, n, total = r.loc[list(names), "MAPE(%)"], r.loc[list(names), "건수"], r.loc["시험용 전체", "MAPE(%)"]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar([f"{names[k]}\n({int(n[k]):,}건)" for k in names], v.values, color=GRAY, width=0.55)
    for i, val in enumerate(v.values):
        ax.text(i, val + 0.3, f"{val:.1f}%", ha="center")
    ax.axhline(total, color=ACCENT, lw=1.2, ls="--")
    ax.text(-0.27, total + 0.3, f"시험용 전체 {total:.1f}%", color=ACCENT, ha="left")
    ax.set_ylim(0, v.max() * 1.25)
    ax.set_ylabel("평균 오차율 (MAPE, %)")
    ax.set_title("권역별 평균 오차율 (시험용)", loc="left", fontsize=11)
    save(fig, "region.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    models()
    actual_vs_pred()
    region()
