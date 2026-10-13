"""EDA 리포트 생성 — 학습용 데이터만 사용 (시험용 지번은 열어보지 않음)

기준: 원래 거래가(price_won). ㎡당 가격·log 같은 타깃 변환은 2번에서 '후보 비교'로만 보여 주고,
적용 여부는 전처리 단계에서 결정 (10/12 사용자 결정)
입력: data/processed/features_all.csv, split.csv / data/raw/trades_rh.csv (거래 유형·매도자: 이상치 해석에만 사용)
출력: reports/eda_report.html (그래프까지 파일 하나에 포함)
실행: python src/eda.py
"""
import base64
import io
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "reports" / "eda_report.html"
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.dpi": 110,
                     "axes.spines.top": False, "axes.spines.right": False})
GU_COLOR = {"강남구": "#d1495b", "강서구": "#00798c", "관악구": "#edae49"}
GUS = list(GU_COLOR)

DESC = {
    "id": "거래 식별자", "gu": "구", "dong": "법정동", "area_m2": "전용면적(㎡)", "floor": "층 (음수 = 지하)",
    "top_floor": "꼭대기층 여부 (1 = 꼭대기)", "building_age": "건물 나이(년, 거래일 기준)", "lat": "위도", "lon": "경도",
    "station_dist_m": "가장 가까운 역까지 직선거리(m)", "n_lines_500m": "반경 500m 안 지하철 노선 수",
    "pub_ppm": "공시가격 ㎡당(원) — 거래일 당시 공개돼 있던 최신 연도(2023~2026)", "pub_level": "공시가격을 찾은 단계 (보조)", "hhld_cnt": "건물 세대수",
    "elevator": "승강기 수", "parking_per_hh": "세대당 주차대수", "structure": "건물 구조",
    "land_per_hh": "세대당 대지면적(㎡)", "t_month": "거래 시점 (2023-10부터 개월 수)",
    "nb_ppm_500m": "주변 ㎡당 시세(원) — 반경 500m·직전 12개월·다른 지번", "nb_cnt_500m": "주변 거래 건수 (보조)",
    "price_won": "정답: 거래가(원)", "deal_date": "거래일",
}
KIND = {
    "연속형 수치": ["area_m2", "building_age", "lat", "lon", "station_dist_m", "pub_ppm", "parking_per_hh", "land_per_hh",
               "nb_ppm_500m", "price_won"],
    "이산형 수치": ["floor", "top_floor", "n_lines_500m", "hhld_cnt", "elevator", "t_month", "nb_cnt_500m"],
    "범주형": ["gu", "dong", "structure", "pub_level"],
    "날짜": ["deal_date"],
    "식별자": ["id"],
}
KIND_OF = {c: k for k, cs in KIND.items() for c in cs}
STRATEGY = {
    "top_floor": "대장이 없어 층수를 모름. 버림 후보라 보류, 쓰게 되면 0으로 대체 + 빈칸 표시 칸",
    "building_age": "대장·실거래 건축년도 둘 다 없거나 거래 뒤 신축. 중앙값 대체 + 빈칸 표시 칸",
    "lat": "좌표를 못 찾은 지번 1개(화곡동 155-43). 학습에서 행 제외, 예측 때는 실행 중 지오코딩",
    "lon": "lat과 같음", "station_dist_m": "lat과 같음", "n_lines_500m": "lat과 같음", "nb_cnt_500m": "lat과 같음",
    "pub_ppm": "그 연도 공시 목록에 없음 (공시 이후 지어진 신축 등). 동 중앙값 대체 + 빈칸 표시 칸",
    "hhld_cnt": "대장 없음 또는 세대수 0 기재. 중앙값 대체 + 빈칸 표시 칸", "parking_per_hh": "hhld_cnt와 같음",
    "elevator": "대장 없음. 0으로 대체(다세대 대부분 승강기 없음) + 빈칸 표시 칸",
    "structure": "대장 없음. '모름' 범주로",
    "land_per_hh": "대장에 대지면적 0 기재. 결측이 많음 → 쓸지부터 결정",
    "nb_ppm_500m": "반경 500m·12개월 안 다른 지번 거래 없음. 반경 1km로 넓혀 재계산 → 그래도 없으면 동 중앙값",
}


def img(fig):
    buf = io.BytesIO()
    fig.tight_layout()  # 여러 칸 그림에서 제목과 눈금 글자가 겹치지 않게
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def table(d, index=False):
    return '<div class="tw">' + d.to_html(index=index, border=0, na_rep="—", escape=False) + "</div>"


def eok(v):
    return f"{v / 1e8:,.2f}억"


# ── 데이터 ──────────────────────────────────────────────
f = pd.read_csv(PROC / "features_all.csv").merge(pd.read_csv(PROC / "split.csv"), on="id")
n_test = (f.set == "test").sum()
feat = f[f.set == "train"].drop(columns="set").reset_index(drop=True)
t = pd.read_csv(ROOT / "data" / "raw" / "trades_rh.csv", dtype=str)
t = t[((t.gu != "강서구") | (t.umdNm == "화곡동")) & (t.houseType == "다세대")
      & (t.cdealType.fillna("").str.strip() != "O")].reset_index(drop=True)
t["id"] = range(1, len(t) + 1)  # make_input.py와 같은 필터·순서 → 같은 id
df = feat.merge(t[["id", "jibun", "dealingGbn", "slerGbn", "mhouseNm"]], on="id")
df["price_eok"] = df.price_won / 1e8  # 단위만 억으로 (그래프 읽기용)
df["pub_man"], df["nb_man"] = df.pub_ppm / 1e4, df.nb_ppm_500m / 1e4  # 단위만 만원으로
df["new"] = df.building_age < 1
df["fl_grp"] = pd.cut(df.floor, [-10, 0, 1, 3, 100], labels=["지하", "1층", "2~3층", "4층 이상"])
gu_med = df.groupby("gu").price_won.median()
CAND = ["area_m2", "floor", "top_floor", "building_age", "station_dist_m", "n_lines_500m", "pub_ppm", "hhld_cnt",
        "elevator", "parking_per_hh", "land_per_hh", "t_month", "nb_ppm_500m", "lat", "lon"]
sec = {}

# ── 0. 컬럼 요약표 ─────────────────────────────────────
summ = pd.DataFrame({
    "컬럼": feat.columns, "뜻": [DESC.get(c, "") for c in feat.columns], "분류": [KIND_OF.get(c, "") for c in feat.columns],
    "데이터 타입": feat.dtypes.astype(str).values, "결측 수": feat.isna().sum().values,
    "결측 비율(%)": (feat.isna().mean() * 100).round(2).values, "고유값 수": feat.nunique().values,
    "메모리(KB)": (feat.memory_usage(deep=True, index=False) / 1024).round(1).values,
})
sec["summary_table"] = table(summ)

# ── 1. 기본 형태 ───────────────────────────────────────
mem_mb = feat.memory_usage(deep=True).sum() / 1024 ** 2
sec["shape"] = (f"<p>학습용 <b>{len(feat):,}행 × {feat.shape[1]}열</b>, 메모리 {mem_mb:.1f}MB. "
                f"(시험용 {n_test:,}행은 분석에서 제외)</p>"
                "<h4>상위 5행 (head)</h4>" + table(feat.head().round(2))
                + "<h4>하위 5행 (tail)</h4>" + table(feat.tail().round(2))
                + "<p class='muted'>컬럼별 데이터 타입·메모리는 맨 위 컬럼 요약표에 있습니다.</p>")


# ── 2. 타깃 분포 ───────────────────────────────────────
def dist_row(name, v):
    q1, q3 = v.quantile([.25, .75])
    out = ((v < q1 - 1.5 * (q3 - q1)) | (v > q3 + 1.5 * (q3 - q1))).sum()
    return [name, f"{v.mean():,.2f}", f"{v.median():,.2f}", round(stats.skew(v), 2), round(stats.kurtosis(v), 2), out]


ppm = df.price_won / df.area_m2  # 2번 '변환 후보 비교'에만 쓰는 임시 계산 (파일·3~6번 분석에는 안 씀)
cands = [("A. 거래가(억) — 원래 값", df.price_eok), ("B. log 거래가", np.log(df.price_won)),
         ("C. ㎡당 가격(만원)", ppm / 1e4), ("D. log ㎡당 가격", np.log(ppm))]
tgt = pd.DataFrame([dist_row(n, v) for n, v in cands], columns=["타깃 후보", "평균", "중앙값", "왜도", "첨도", "IQR 밖 건수"])
fig, ax = plt.subplots(2, 2, figsize=(11, 6.5))
for a, (title, v) in zip(ax.flat, cands):
    for g in GUS:
        a.hist(v[df.gu == g], bins=60, alpha=.55, color=GU_COLOR[g], label=g)
    a.set_title(f"{title}  (왜도 {stats.skew(v):.2f})")
ax[0, 0].legend()
opts = pd.DataFrame([
    ["A. 거래가 그대로", tgt.iloc[0, 3], "과제 출력(원)과 같아서 되돌리는 계산이 필요 없음", "크게 치우침. 몇 안 되는 고가 거래가 선형회귀 결과를 크게 움직일 수 있음"],
    ["B. log 거래가", tgt.iloc[1, 3], "치우침이 줄어듦. 계수를 '몇 % 비싸다'로 읽을 수 있음", "면적 효과는 그대로 섞여 있음 (5-5 사례)"],
    ["C. ㎡당 가격", tgt.iloc[2, 3], "면적 차이를 뺀 비교가 됨", "여전히 치우침. 예측 후 × 면적으로 되돌려야 함"],
    ["D. log ㎡당 가격", tgt.iloc[3, 3], "네 후보 중 가장 대칭", "변환이 두 번이라 설명이 한 단계 더 필요"],
], columns=["후보", "왜도", "장점", "단점"])
sec["target"] = (img(fig) + table(tgt) + f"""
<p class="muted">왜도 = 분포가 한쪽으로 치우친 정도 (0이면 좌우 대칭). 첨도 = 꼬리가 두꺼운 정도 (0이면 정규분포 수준). IQR 밖 = 상자그림 기준 이상치 후보.</p>
<div class="callout"><b>처리 방향 후보 — 결정은 전처리 단계에서</b>
<p>원래 거래가는 왜도 {tgt.iloc[0, 3]}로 오른쪽(고가) 꼬리가 깁니다. 권역별 거래가 중앙값은 {", ".join(f"{g} {eok(gu_med[g])}" for g in GUS)}로 수준 차이가 큽니다.
아래는 바꿀 수 있는 후보의 비교이며, <b>이 리포트의 3~6번은 모두 A(원래 거래가) 기준</b>입니다.</p>{table(opts)}</div>""")

# ── 3. 데이터 타입별 분류 ───────────────────────────────
cls = pd.DataFrame([[k, ", ".join(f"<code>{c}</code>" for c in cs if c in feat.columns)] for k, cs in KIND.items()],
                   columns=["분류", "컬럼"])
sec["types"] = table(cls) + """<p class="muted">텍스트형 컬럼은 없습니다 (건물명은 가격 요인이 아니고 외우기 위험이 있어 후보에서 뺌, 결정기록 4번).
이산형 중 <code>top_floor</code>는 0/1 두 값만 갖는 이진형입니다. 범주형 중 <code>dong</code>은 고유값이 많아 모델에 넣을 때 표본 적은 동을 묶어야 할 수 있습니다.</p>"""

# ── 4. 결측값·중복값 ───────────────────────────────────
miss = summ[summ["결측 수"] > 0][["컬럼", "뜻", "결측 수", "결측 비율(%)"]].copy()
miss["처리 전략"] = miss["컬럼"].map(STRATEGY)
exact_dup = feat.drop(columns="id").duplicated().sum()
dkey = ["jibun", "dong", "floor", "area_m2", "deal_date", "price_won"]
bulk = df[df.duplicated(dkey, keep=False)]
n_bulk_grp = bulk.groupby(dkey).ngroups
sec["missing"] = (table(miss) + f"""
<p class="muted">"빈칸 표시 칸" = 원래 빈칸이었는지 0/1로 남기는 칸. 대체값을 넣어도 모델이 "이건 추정값"이라는 걸 알 수 있게 함.</p>
<h4>중복값</h4><ul>
<li>id를 뺀 모든 칸이 같은 행(완전 중복): <b>{exact_dup}건</b></li>
<li>같은 지번·층·면적·거래일·가격 묶음: <b>{n_bulk_grp}묶음 {len(bulk)}건</b> (직거래 {(bulk.dealingGbn == "직거래").sum()}건, 신축 1년 미만 {bulk.new.sum()}건)<br>
→ 같은 층에 같은 면적의 호가 여러 개라 <b>서로 다른 호가 같은 날 함께 팔린 것</b>(일괄 매각)으로 보임 (추정). 데이터 오류가 아니라 지우지 않음. 다만 여러 호를 한꺼번에 사면 개별 시세보다 싸게 거래될 수 있어 6번 이상치 표에서 다룸</li></ul>""")

# ── 5. 관계 분석 (타깃 = 원래 거래가) ───────────────────
methods = pd.DataFrame([
    ["산점도", "A가 연속형", "관계 모양(직선·곡선), 점이 몰린 구간", "점이 수천 개면 겹쳐서 경향이 안 보임. 다른 변수 효과가 섞인 채로 보임"],
    ["구간별 중앙값 선 그래프", "A가 연속형, 표본이 많을 때", "구간마다 B의 대표값 → 방향과 꺾이는 지점", "구간을 나누는 방식에 따라 모양이 바뀜. 구간마다 표본 수가 다르면 믿을 수 있는 정도도 다름"],
    ["그룹별 상자그림 + 크러스컬-월리스 검정", "A가 범주형 (구, 구조, 층 그룹)", "그룹 간 중앙값·퍼짐 차이, 그 차이가 우연인지", "표본이 많으면 아주 작은 차이도 '유의미'로 나옴 → 차이의 크기는 따로 봐야 함. 교란 변수는 통제 못 함"],
    ["상관계수 (피어슨·스피어만)", "A, B 둘 다 수치형", "관계의 세기를 숫자 하나로 (-1~1)", "피어슨은 직선 관계만, 스피어만은 한 방향 관계만 잡음 (U자형은 0에 가깝게 나옴). 상관은 원인이 아님. 집단 차이가 만든 착시가 섞임 (5-5 경도 사례)"],
    ["층화 비교 (권역 안에서 따로 계산)", "집단 간 수준 차이가 클 때", "착시를 걸러내고, 권역마다 방향이 같은지 확인", "권역 안 표본이 줄어 흔들림. 권역 말고 다른 교란은 그대로 남음"],
    ["통제 비교 (교란 변수 구간 안에서 비교)", "A와 같이 움직이는 변수가 의심될 때", "A만의 효과에 가까운 값 (5-5 건물 나이 사례)", "무엇을 통제할지 미리 알아야 함. 구간을 잘게 나누면 표본 부족"],
    ["다중회귀 계수 (R의 lm)", "여러 변수를 동시에 볼 때", "다른 변수를 고정했을 때 A가 B를 얼마나 바꾸는지", "변수끼리 많이 겹치면(다중공선성) 계수가 불안정. 직선 관계를 가정. 모델에 없는 변수의 효과는 여전히 섞임"],
], columns=["방법", "언제 쓰나", "알 수 있는 것", "한계"])

corr_cols = CAND + ["price_won"]
cm = df[corr_cols].corr("spearman")
fig, ax = plt.subplots(figsize=(10, 8.5))
im = ax.imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_cols)), corr_cols, rotation=60, ha="right")
ax.set_yticks(range(len(corr_cols)), corr_cols)
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        ax.text(j, i, f"{cm.iat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                color="white" if abs(cm.iat[i, j]) > .6 else "black")
fig.colorbar(im, shrink=.7)
ax.set_title("스피어만 상관 (학습용)")
heat = img(fig)
pairs = [(a, b, cm.loc[a, b]) for i, a in enumerate(CAND) for b in CAND[i + 1:] if abs(cm.loc[a, b]) >= .6]
pairs_html = "".join(f"<li><code>{a}</code> ↔ <code>{b}</code>: {r:.2f}</li>" for a, b, r in sorted(pairs, key=lambda x: -abs(x[2])))


def vif(d):
    """변수 하나를 나머지 전부로 맞히는 회귀의 R² → VIF = 1 / (1 − R²). R의 car::vif와 같은 원리"""
    X = d.to_numpy(float)
    out = []
    for j in range(X.shape[1]):
        y, others = X[:, j], np.column_stack([np.ones(len(X)), np.delete(X, j, axis=1)])
        beta = np.linalg.lstsq(others, y, rcond=None)[0]
        r2 = 1 - ((y - others @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        out.append((r2, 1 / (1 - r2)))
    return out


vif_data = df[CAND].dropna()  # 빈칸이 하나라도 있는 행은 제외 (세대당 대지 빈칸 13.8% 때문에 줄어듦)
vif_rows = []
for c, (r2, v) in zip(CAND, vif(vif_data)):
    partner = cm.loc[c, [x for x in CAND if x != c]].abs().idxmax()
    level = ('<span class="bad">높음 (10 이상)</span>' if v >= 10 else '<span class="mid">주의 (5~10)</span>' if v >= 5
             else '<span class="ok">낮음 (5 미만)</span>')
    vif_rows.append([f"<code>{c}</code>", DESC[c], round(v, 2), f"{r2 * 100:.0f}%", level,
                     f"<code>{partner}</code> ({cm.loc[c, partner]:.2f})"])
vif_tbl = pd.DataFrame(vif_rows, columns=["변수", "뜻", "VIF", "나머지 변수로 설명되는 비율(R²)", "판정", "가장 많이 겹치는 상대 (스피어만 상관)"])
vif_tbl = vif_tbl.sort_values("VIF", ascending=False)

rgu = {c: {g: df[df.gu == g][[c, "price_won"]].corr("spearman").iloc[0, 1] for g in GUS} for c in CAND}
NOTE = {  # 상관 판정에 덧붙이는 해석 (자동 판정만으로 남기거나 버리면 안 되는 경우)
    "t_month": "시점수정은 감정평가 필수 요소 → 상관이 약해도 남길지 따로 판단 (5-4 월별 추세)",
    "lat": "위경도를 직선으로 넣으면 권역마다 방향이 다름 → 위치 효과는 동·주변 시세로 반영할지 검토",
    "lon": "lat과 같음 (5-5 착시 사례)",
    "building_age": f"강남은 {rgu['building_age']['강남구']:.2f}로 거의 0 → 오래된 건물이 더 넓어서 효과가 가려짐 (5-5 교란 사례)",
    "area_m2": f"넓을수록 비쌈. 세대당 대지({cm.loc['area_m2', 'land_per_hh']:.2f})·건물 나이와 겹침",
    "land_per_hh": f"면적과 상관 {cm.loc['area_m2', 'land_per_hh']:.2f}로 크게 겹침",
    "elevator": f"건물 나이와 상관 {cm.loc['building_age', 'elevator']:.2f}로 같이 움직임",
    "hhld_cnt": f"강남만 반대 방향({rgu['hhld_cnt']['강남구']:.2f})",
    "station_dist_m": f"강남만 반대 방향({rgu['station_dist_m']['강남구']:+.2f})",
    "pub_ppm": "공시가격은 ㎡당 값이라 면적 차이가 빠져 있어 총액과의 상관이 낮아질 수 있음",
    "nb_ppm_500m": "pub_ppm과 같음 (㎡당 값)",
}


def verdict(rs):
    same = len(set(np.sign(rs))) == 1 and np.sign(rs[0]) != 0
    if same and (np.abs(rs) >= .1).all():
        return '<span class="ok">일관 → 남김 후보</span>'
    if same:
        return '<span class="mid">방향 같음·약함 → 보류</span>'
    return '<span class="bad">방향 엇갈림 → 버림 후보</span>'


rows = []
for c in CAND:
    rs = np.array([rgu[c][g] for g in GUS])
    rows.append([f"<code>{c}</code>", DESC[c], round(cm.loc[c, "price_won"], 2), *rs.round(2), verdict(rs), NOTE.get(c, "")])
rel = pd.DataFrame(rows, columns=["변수", "뜻", "전체", *GUS, "자동 판정", "해석"])
rel = rel.reindex(rel["전체"].abs().sort_values(ascending=False).index)

# 구간별 거래가 중앙값 (권역별)
panels = [("building_age", "건물 나이(년)"), ("area_m2", "전용면적(㎡)"), ("fl_grp", "층 그룹"),
          ("station_dist_m", "역거리(m)"), ("pub_man", "공시가격 ㎡당(만원)"), ("nb_man", "주변 ㎡당 시세(만원)"),
          ("hhld_cnt", "세대수"), ("elevator", "승강기 수"), ("land_per_hh", "세대당 대지(㎡)")]
fig, axes = plt.subplots(3, 3, figsize=(13, 10.5))
for a, (c, title) in zip(axes.flat, panels):
    for g in GUS:
        d = df[df.gu == g]
        if c == "fl_grp":
            m = d.groupby("fl_grp", observed=True).price_eok.median()
            a.plot(range(len(m)), m.values, "o-", color=GU_COLOR[g], label=g)
            a.set_xticks(range(len(m)), m.index)
        elif c == "elevator":
            m = d.groupby(d.elevator.clip(upper=2)).price_eok.median()
            a.plot(m.index, m.values, "o-", color=GU_COLOR[g], label=g)
            a.set_xticks([0, 1, 2], ["0", "1", "2+"])
        else:
            bins = pd.qcut(df[c], 6, duplicates="drop")  # 전체 기준 6구간 → 권역끼리 같은 구간으로 비교
            m = d.groupby(bins[d.index], observed=True).agg(x=(c, "median"), y=("price_eok", "median"))
            a.plot(m.x, m.y, "o-", color=GU_COLOR[g], label=g)
    a.set_title(title)
axes[0, 0].set_ylabel("거래가 중앙값(억)")
axes[0, 0].legend()
binned = img(fig)

# 월별 추세
df["ym"] = pd.to_datetime(df.deal_date).dt.to_period("M").dt.to_timestamp()
mt = df.groupby(["ym", "gu"]).price_eok.median().unstack().rolling(3, min_periods=2).median()
fig, ax = plt.subplots(figsize=(11, 3.8))
for g in GUS:
    ax.plot(mt.index, mt[g], color=GU_COLOR[g], label=g, lw=2)
ax.set_ylabel("거래가 중앙값(억), 3개월 이동")
ax.legend()
trend = img(fig)
first, last = mt.dropna().iloc[0], mt.dropna().iloc[-1]

# 착시·교란 사례
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ycap = df.price_eok.quantile(.99)
for g in GUS:
    d = df[df.gu == g]
    ax[0].scatter(d.lon, d.price_eok, s=4, alpha=.35, color=GU_COLOR[g], label=g)
ax[0].set_ylim(0, ycap)
ax[0].set_xlabel("경도 (동쪽일수록 큼)")
ax[0].set_ylabel("거래가(억)")
ax[0].set_title(f"경도: 전체 상관 {cm.loc['lon', 'price_won']:.2f}, 권역 안에서는 엇갈림")
ax[0].legend(markerscale=4)
gn = df[df.gu == "강남구"].copy()
age_lb = ["0~5년", "5~10년", "10~20년", "20~30년", "30년+"]
gn["age_b"] = pd.cut(gn.building_age, [0, 5, 10, 20, 30, 60], right=False, labels=age_lb)
gn["area_b"] = pd.cut(gn.area_m2, [0, 30, 45, 60, 300], labels=["30㎡ 미만", "30~45㎡", "45~60㎡", "60㎡ 이상"])
cell = gn.groupby(["area_b", "age_b"], observed=True).price_eok.agg(["median", "size"])
cell_med = cell["median"].where(cell["size"] >= 30).unstack()  # 30건 미만 칸은 그리지 않음 (표본 부족)
overall_age = gn.groupby("age_b", observed=True).agg(p=("price_eok", "median"), a=("area_m2", "median"))
ax[1].plot(age_lb, overall_age.p.values, "k--o", lw=2, label="강남 전체 (면적 섞임)")
for (ab, row), col in zip(cell_med.iterrows(), ["#f4a261", "#e76f51", "#2a9d8f", "#264653"]):
    ax[1].plot(age_lb, row.reindex(age_lb).values, "o-", color=col, label=f"면적 {ab}")
ax[1].set_ylabel("거래가 중앙값(억)")
ax[1].set_title("강남: 면적 구간을 고정하면 오래될수록 싸짐 (30건 미만 칸 생략)")
ax[1].legend(fontsize=8)
confound = img(fig)
r0 = cell_med.iloc[0].dropna()

sec["relation"] = (f"""
<p class="callout">이 섹션의 B(타깃)는 <b>원래 거래가</b>입니다. 변환(㎡당·log)은 적용하지 않았습니다.</p>
<h3 id="s5-1">5-1. "A가 B에 영향을 준다"를 확인하는 방법과 한계</h3>
<p>한 가지 방법만으로는 결론을 내리지 않고, 아래 순서로 좁혀 갑니다:
<b>상관계수로 후보 고르기 → 층화 비교로 착시 거르기 → 통제 비교로 교란 확인 → 다중회귀로 최종 확인(5단계 모델링)</b>.</p>
{table(methods)}
<h3 id="s5-2">5-2. 변수끼리 상관 (히트맵)</h3>
<p class="why">차트 선택 이유: 변수가 16개면 짝이 120개라 표로는 못 봄. 색으로 겹치는 짝을 빨리 찾기 위해 히트맵을 씀. 순위 기반인 스피어만을 써서 극단값(고가 거래)에 덜 흔들리게 함.</p>
{heat}
<p><b>많이 겹치는 짝 (|상관| 0.6 이상)</b> — 둘 다 넣으면 선형회귀 계수가 불안정해질 수 있음:</p><ul>{pairs_html}</ul>
<h3 id="s5-2b">5-2b. VIF (다중공선성 진단)</h3>
<p class="why">VIF = 1 ÷ (1 − R²). R²는 "이 변수를 나머지 변수 <b>전부</b>로 얼마나 맞힐 수 있나"(0~100%). 짝끼리 상관(히트맵)으로는 안 보이는 세 변수 이상의 겹침도 잡음.
흔히 5 이상이면 주의, 10 이상이면 문제로 보지만 <b>관례 기준이지 절대 기준은 아님</b>. 타깃과 관계없이 변수들끼리만 계산하므로, 타깃을 정하기 전에도 결과가 같음.</p>
{table(vif_tbl)}
<p class="muted">계산에 쓴 행: 빈칸이 없는 {len(vif_data):,}건 (학습용 {len(df):,}건 중). 범주형(구·동·구조)은 빠져 있음.
VIF는 직선 관계(피어슨 계열) 기준이고 히트맵은 순위(스피어만) 기준이라, 숫자 크기가 딱 맞아떨어지지는 않음.
<b>제거는 하지 않았음</b> — 높은 변수가 있으면 "가장 많이 겹치는 상대"와 짝으로 보고 하나를 고른 뒤 VIF를 다시 계산해 확인.</p>
<h3 id="s5-3">5-3. 타깃(거래가)과의 상관: 전체 vs 권역 안</h3>
<p class="why">판정 규칙: 3개 권역에서 방향(부호)이 같고 모두 |0.1| 이상이면 "일관". 방향이 엇갈리면 전체 상관은 권역 차이가 만든 착시일 수 있음. (연립 제외 결정 때 쓴 논리와 같음)</p>
{table(rel)}
<p class="muted">자동 판정은 출발점입니다. 해석 칸처럼 부동산 논리나 교란을 같이 보고 최종 결정합니다.</p>
<h3 id="s5-4">5-4. 구간별 거래가 중앙값 그래프 (권역별)</h3>
<p class="why">차트 선택 이유: 점 8,000개 산점도보다 경향이 잘 보이고, 중앙값이라 고가 거래에 덜 흔들림. 권역마다 선을 따로 그려 방향이 같은지 비교.
가로축 구간은 전체 데이터 기준 6등분이라 권역끼리 같은 구간으로 비교됨.</p>
{binned}
<p class="why">월별 추세 (시점수정 근거): 3개월 이동 중앙값으로 월별 들쭉날쭉함을 줄임.</p>
{trend}
<p>처음 → 마지막 3개월 중앙값: {", ".join(f"{g} {first[g]:.2f} → {last[g]:.2f}억 ({(last[g] / first[g] - 1) * 100:+.0f}%)" for g in GUS)}.
<b>한계:</b> 총액 추세는 그달에 팔린 집의 면적 구성에 따라 흔들림 (넓은 집이 많이 팔린 달은 올라 보임).</p>
<h3 id="s5-5">5-5. 착시·교란 사례 2개 (방법의 한계가 실제로 나타난 곳)</h3>
{confound}
<ul>
<li><b>경도 (착시):</b> 전체 상관은 {cm.loc['lon', 'price_won']:.2f}로 "동쪽일수록 비싸다"처럼 보이지만, 권역 안에서는
{", ".join(f"{g} {rgu['lon'][g]:.2f}" for g in GUS)}로 엇갈림. 강남이 동쪽에 있고 비싸서 생긴 착시 → <b>상관계수의 한계, 층화 비교로 걸러냄</b>
(왼쪽 그림은 상위 1%인 {ycap:.0f}억 초과 거래를 잘라서 표시)</li>
<li><b>강남 건물 나이 (교란):</b> 강남 전체로는 0~5년 {overall_age.p.iloc[0]:.2f}억, 30년+ {overall_age.p.iloc[-1]:.2f}억으로 나이 효과가 거의 안 보임(상관 {rgu['building_age']['강남구']:.2f}).
오래된 건물의 면적 중앙값이 {overall_age.a.iloc[-1]:.0f}㎡로 신축({overall_age.a.iloc[0]:.0f}㎡)보다 넓어서, "오래돼서 싸지는 효과"와 "넓어서 비싸지는 효과"가 서로 상쇄됨.
면적 구간을 고정하면 {cell_med.index[0]}에서 {r0.index[0]} {r0.iloc[0]:.2f}억 → {r0.index[-1]} {r0.iloc[-1]:.2f}억처럼 오래될수록 싸짐 → <b>통제 비교로 확인한 교란</b>.
총액을 타깃으로 쓸 경우 건물 나이 효과는 면적과 함께 넣어야 드러남</li>
</ul>""")

# ── 6. 이상치 (거래가 기준) ─────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
box_vars = [("price_eok", "거래가(억)"), ("area_m2", "전용면적(㎡)"), ("floor", "층"), ("building_age", "건물 나이"),
            ("station_dist_m", "역거리(m)"), ("pub_man", "공시가격 ㎡당(만원)"), ("hhld_cnt", "세대수"), ("nb_man", "주변 ㎡당 시세(만원)")]
for a, (c, title) in zip(axes.flat, box_vars):
    a.boxplot([df[df.gu == g][c].dropna() for g in GUS], tick_labels=GUS, flierprops=dict(markersize=2, alpha=.4))
    a.set_title(title)
boxes = img(fig)

df["kind"] = np.select([df.dealingGbn == "중개거래", df.new], ["중개거래", "직거래(신축 1년 미만)"], "직거래(기존 건물)")
kind_tbl = df.groupby(["gu", "kind"]).agg(건수=("price_won", "size"), 거래가_중앙값=("price_won", lambda s: eok(s.median())),
                                          면적_중앙값=("area_m2", lambda s: f"{s.median():.1f}㎡")).reset_index()
kind_tbl.columns = ["권역", "거래 유형", "건수", "거래가 중앙값", "면적 중앙값"]

fence = {}
for g in GUS:
    v = df[df.gu == g].price_won
    q1, q3 = v.quantile([.25, .75])
    fence[g] = (q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1))
iqr_hi = df[[p > fence[g][1] for g, p in zip(df.gu, df.price_won)]]
iqr_lo = df[[p < fence[g][0] for g, p in zip(df.gu, df.price_won)]]
low1 = df[df.price_won <= df.groupby("gu").price_won.transform(lambda s: s.quantile(.01))]
big = df.nlargest(1, "price_won").iloc[0]
tall = df[df.floor >= 10]
new_direct = df[df.kind == "직거래(신축 1년 미만)"]
under = df[df.floor < 1]
fl_med = df.groupby("fl_grp", observed=True).agg(p=("price_won", "median"), a=("area_m2", "median"))
cases = pd.DataFrame([
    ["최고가 1건", 1, f"{big.dong} {big.jibun} 지하 {big.area_m2}㎡ {eok(big.price_won)} — {big.dong} 거래가 중앙값 {eok(df[df.dong == big.dong].price_won.median())}의 "
                   f"{big.price_won / df[df.dong == big.dong].price_won.median():.0f}배, 강남 상위 1% 경계 {eok(df[df.gu == '강남구'].price_won.quantile(.99))}의 "
                   f"{big.price_won / df[df.gu == '강남구'].price_won.quantile(.99):.1f}배",
     "강남 고급 빌라의 넓은 지하층", "지하인데 같은 동 중앙값의 8배 → 상가·업무용이거나 입력 오류일 가능성", "<b>결정 필요</b>: 제외 vs 유지"],
    ["권역별 IQR 위쪽", len(iqr_hi), ", ".join(f"{g} {(iqr_hi.gu == g).sum()}건(경계 {eok(fence[g][1])})" for g in GUS),
     "넓은 집·신축·고급 빌라 → 진짜 가격", "—", "유지. 모델을 만든 뒤 오차가 큰 건만 다시 확인"],
    ["10층 이상", len(tall), f"{tall.mhouseNm.iloc[0]} 등, 모두 승강기 있음·신축 10년 이내",
     "실제 거래. 도시형생활주택 등 법적 다세대 기준(주택 층수 4개 이하)과 다른 건물로 보임(추정)", "층수 기재 오류는 아님(같은 건물 여러 건)",
     "유지. 층 변수를 그대로 쓸지 묶을지는 전처리에서"],
    ["지하층", len(under), f"거래가 중앙값 지하 {eok(fl_med.p['지하'])} vs 1층 {eok(fl_med.p['1층'])} (면적 중앙값 {fl_med.a['지하']:.0f}㎡ vs {fl_med.a['1층']:.0f}㎡로 비슷)",
     "면적이 비슷한데 더 쌈 → 지하 할인은 실제 시장 현상", "—", "유지 + 지하 여부를 변수로"],
    ["신축 직거래", len(new_direct), "아래 거래 유형 표: 같은 권역 중개거래보다 거래가가 높고 면적은 작음",
     "건축주가 직접 분양한 거래로 보임(추정). 신축 프리미엄이 붙은 실제 시장 가격", "분양가에는 마케팅·옵션이 섞여 기존 건물 시세와 다를 수 있음",
     "<b>결정 필요</b>: 유지 vs 신축 여부 변수 추가 vs 제외"],
    ["같은 날 일괄 매각 묶음", len(bulk), f"{n_bulk_grp}묶음, 예: 화곡동 982-6 여러 호가 같은 날 1.4억씩",
     "여러 호를 한꺼번에 산 거래", "개별 시세보다 싸게 거래됐을 수 있음", "<b>결정 필요</b>: 유지 vs 묶음당 1건만 남기기"],
    ["권역별 거래가 하위 1%", len(low1), f"그중 직거래 {(low1.dealingGbn == '직거래').sum()}건, 지하 {(low1.floor < 1).sum()}건, 최저 {eok(low1.price_won.min())}",
     "지하·초소형·노후 건물이면 실제로 쌀 수 있음", "가족 간 거래 등 특수 사정 (사정보정 대상)", "<b>결정 필요</b>: 그중 직거래만 제외 vs 유지"],
], columns=["이상치 후보", "건수", "내용", "비즈니스 해석 (진짜 가격)", "노이즈 해석 (시세가 아님)", "처리 제안"])
sec["outlier"] = (f"""
<p class="why">차트 선택 이유: 상자그림은 사분위(IQR) 기준으로 이상치를 점으로 보여 줘서 권역별로 비교하기 좋음. 단, "상자 밖 = 버릴 값"이 아님 →
아래 표처럼 하나씩 <b>진짜 가격인지(비즈니스)</b>, <b>시세가 아닌 값인지(노이즈)</b> 따져서 결정.</p>
{boxes}
<p>거래가 상자그림에서 IQR 밖 점은 대부분 <b>위쪽(고가)</b>입니다 (위 {len(iqr_hi)}건, 아래 {len(iqr_lo)}건). 원래 거래가의 꼬리가 오른쪽으로 길기 때문 (2번 왜도).</p>
<h4>거래 유형별 비교 (사정보정 검토용)</h4>
{table(kind_tbl)}
<p class="muted">거래 유형·매도자는 예측 때 알 수 없어 변수로는 못 쓰고(결정기록 4번), 이상치를 해석하는 데만 씀. 면적이 다르면 거래가도 달라지므로 면적 중앙값을 함께 표시.
면적 차이를 뺀 비교(㎡당, 주변 시세 대비)는 타깃 변환을 정하는 전처리 단계에서 필요하면 다시 봄.</p>
{table(cases)}""")

# ── 7. 종합 ────────────────────────────────────────────
keep = rel[rel["자동 판정"].str.contains("남김")]["변수"].str.replace(r"</?code>", "", regex=True).tolist()
hold = rel[rel["자동 판정"].str.contains("보류")]["변수"].str.replace(r"</?code>", "", regex=True).tolist()
drop = rel[rel["자동 판정"].str.contains("버림")]["변수"].str.replace(r"</?code>", "", regex=True).tolist()
sec["wrapup"] = f"""
<h3>전처리에서 결정할 것 (우선순위 순)</h3><ol>
<li><b>이상치 4개:</b> 최고가 1건 / 신축 직거래 {len(new_direct)}건 / 일괄 매각 {len(bulk)}건 / 권역 하위 1% 중 직거래 {(low1.dealingGbn == '직거래').sum()}건 (6번 표)</li>
<li><b>타깃을 바꿀지:</b> 원래 거래가는 왜도 {tgt.iloc[0, 3]}로 치우침. 후보 4개 비교는 2번 표.
5-5처럼 총액 기준에서는 면적 효과가 다른 효과(건물 나이)를 가릴 수 있다는 점도 판단 재료</li>
<li><b>변수 (거래가 기준 자동 판정):</b> 남김 후보 {len(keep)}개({", ".join(keep)}), 보류 {len(hold)}개({", ".join(hold)}), 버림 후보 {len(drop)}개({", ".join(drop)}).
자동 판정은 타깃을 무엇으로 정하느냐에 따라 바뀔 수 있어서 타깃 결정 뒤 다시 확인</li>
<li><b>겹치는 변수:</b> 면적 ↔ 세대당 대지({cm.loc['area_m2', 'land_per_hh']:.2f}), 건물 나이 ↔ 승강기({cm.loc['building_age', 'elevator']:.2f}), 역거리 ↔ 노선 수({cm.loc['station_dist_m', 'n_lines_500m']:.2f}).
VIF 10 이상: {", ".join(vif_tbl[vif_tbl.VIF >= 10]["변수"].str.replace(r"</?code>", "", regex=True)) or "없음"} / 5~10: {", ".join(vif_tbl[(vif_tbl.VIF >= 5) & (vif_tbl.VIF < 10)]["변수"].str.replace(r"</?code>", "", regex=True)) or "없음"} (5-2b). 짝마다 하나만 남길지 결정</li>
<li><b>결측:</b> 모두 2% 이하(세대당 대지 13.8% 제외). 4번 표의 처리 전략 중 선택</li></ol>
<h3>모델링 때 주의할 점</h3><ul>
<li>권역 간 거래가 수준 차이가 큼({", ".join(f"{g} {eok(gu_med[g])}" for g in GUS)}) → 권역(또는 동) 변수 필수. 권역마다 시점 추세도 다름</li>
<li>자체 검증은 회사 검증과 같은 조건으로: 시험용 지번 중 <b>최근 12개월 거래</b>로 오차 측정</li></ul>"""

# ── HTML 조립 ──────────────────────────────────────────
findings = f"""<ol>
<li><b>기준:</b> 이 리포트의 관계·이상치 분석은 모두 <b>원래 거래가</b> 기준 (변환 없음)</li>
<li><b>권역 차이가 큼:</b> 거래가 중앙값 {", ".join(f"{g} {eok(gu_med[g])}" for g in GUS)}</li>
<li><b>거래가는 오른쪽으로 치우침</b> (왜도 {tgt.iloc[0, 3]}): 변환 후보 4개 비교는 2번, 결정은 전처리에서</li>
<li><b>권역 안에서도 방향이 일관된 변수:</b> {", ".join(keep)}</li>
<li><b>착시·교란:</b> 경도는 권역을 고정하면 방향이 엇갈림(착시). 강남 건물 나이 효과는 면적에 가려짐(교란)</li></ol>
<p><b>결정 필요:</b> 이상치 4개(6번), 타깃 변환 여부(2번), 변수 남김·버림(5-3)</p>"""
nav = [("summary", "컬럼 요약표"), ("s1", "1. 기본 형태"), ("s2", "2. 타깃 분포"), ("s3", "3. 타입 분류"), ("s4", "4. 결측·중복"),
       ("s5", "5. 관계 분석"), ("s6", "6. 이상치"), ("s7", "7. 종합")]
html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>다세대 시세 EDA</title><style>
:root{{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1f;--muted:#6b6b70;--line:#e3e3e0;--accent:#00798c;--soft:#eef6f7;--warn:#fff6e5}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#161618;--card:#1f1f22;--ink:#ececec;--muted:#a0a0a8;--line:#333338;--accent:#4fb3c4;--soft:#1d2a2d;--warn:#2d2718}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 "Malgun Gothic","Apple SD Gothic Neo",sans-serif}}
.wrap{{max-width:1180px;margin:0 auto;padding:24px 16px 80px}}
header h1{{margin:0 0 4px;font-size:26px}} header p{{margin:0;color:var(--muted)}}
nav{{position:sticky;top:0;background:var(--bg);padding:10px 0;margin:16px 0;border-bottom:1px solid var(--line);display:flex;flex-wrap:wrap;gap:6px 14px;z-index:5}}
nav a{{color:var(--accent);text-decoration:none;font-size:14px}}
section{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:20px 22px;margin:18px 0}}
h2{{margin:0 0 12px;font-size:20px}} h3{{margin:26px 0 8px;font-size:17px}} h4{{margin:18px 0 6px}}
.tw{{overflow-x:auto;margin:10px 0}} table{{border-collapse:collapse;font-size:13px;width:100%}}
th,td{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}} th{{background:var(--soft);white-space:nowrap}}
img{{max-width:100%;height:auto;display:block;margin:10px 0;border-radius:6px;background:#fff}}
.muted{{color:var(--muted);font-size:13px}} .why{{background:var(--soft);padding:8px 12px;border-radius:6px;font-size:14px}}
.callout{{background:var(--warn);padding:10px 14px;border-radius:6px}} code{{font-size:12.5px;background:var(--soft);padding:1px 4px;border-radius:3px}}
.ok{{color:#2e7d32;font-weight:bold}} .mid{{color:#b7791f;font-weight:bold}} .bad{{color:#c62828;font-weight:bold}}
</style></head><body><div class="wrap">
<header><h1>다세대 시세 EDA 리포트</h1>
<p>화곡동·강남구·관악구 다세대 실거래 (2023-10 ~ 2026-10) · 학습용 {len(feat):,}건만 분석 (시험용 {n_test:,}건은 열어보지 않음) · 기준: 원래 거래가 · 생성 {date.today()} · <code>python src/eda.py</code></p></header>
<nav>{"".join(f'<a href="#{k}">{v}</a>' for k, v in nav)}</nav>
<section><h2>한눈에 보기</h2>{findings}</section>
<section id="summary"><h2>컬럼 요약표 (데이터 타입 · 결측 · 고유값)</h2>{sec["summary_table"]}</section>
<section id="s1"><h2>1. 기본 형태</h2>{sec["shape"]}</section>
<section id="s2"><h2>2. 타깃 분포</h2>{sec["target"]}</section>
<section id="s3"><h2>3. 데이터 타입별 분류</h2>{sec["types"]}</section>
<section id="s4"><h2>4. 결측값 · 중복값</h2>{sec["missing"]}</section>
<section id="s5"><h2>5. 주요 변수 간 관계</h2>{sec["relation"]}</section>
<section id="s6"><h2>6. 이상치</h2>{sec["outlier"]}</section>
<section id="s7"><h2>7. 종합 요약 · 모델링 준비</h2>{sec["wrapup"]}</section>
</div></body></html>"""
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(html, encoding="utf-8")
print(f"저장: {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:,.0f}KB)")
