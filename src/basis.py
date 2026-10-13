"""basis 문장 (산출 근거 한 줄) — 템플릿 (결정기록.md 15번)

내용 (10/13 사용자 결정): 사실만 — 추정가가 주변 기준가보다 몇 %, 공시가격의 몇 배, 건물 특징, 신뢰도와 그 이유. 모델 요인 분해(SHAP)는 안 씀
주변 기준가 (10/13 사용자 결정): 반경 500m·최근 12개월·다른 지번 중 비슷한 면적(전용 ±20%) 거래의 ㎡당 중앙값 × 면적.
  3건 미만이면 면적 구분 없는 주변 거래(모델의 nb_ppm_500m)로 돌아감
최종 방식 (10/13 사용자 결정): 템플릿만. 숫자는 모델·데이터 값 그대로 → 0원, 평가자 API 키 불필요, 숫자 오류 없음
  Haiku 5.5 다듬기 시험 결과(reports/basis_haiku_test.html, 이전 템플릿 문구로 10건): 숫자 검사 통과 9/10이었지만
  통과한 9건 중 3건이 숫자는 그대로 두고 가리키는 대상을 바꿈 (예: 추정가÷공시가격 2.80배를 주변 기준가÷공시가격으로 서술,
  실제는 2.54배) → 숫자 검사로는 못 잡는 오류라 템플릿 채택, 템플릿에 주어(추정가)와 기준가 계산식을 넣어 뜻을 분명히 함
  Haiku 코드는 기록·재시험용으로 남김 (기본 꺼짐). 다듬기 → 숫자 검사(부호·소수점까지 같은 묶음) → 불일치·키 없음·호출 실패면 템플릿
Haiku 설정: effort low, max_tokens 1024 (Haiku 5.5는 생각이 기본으로 켜져 있어 그 몫까지), temperature 등은 넣지 않음 (400 오류)
predict.py용 함수: template(facts) — make_basis(facts, client)는 client 없이 부르면 템플릿과 같음
실행: python src/basis.py         (템플릿 시험, API 호출 없음 → reports/basis_samples.html)
      python src/basis.py haiku   (Haiku 비교 재시험, API 호출·비용 발생 → reports/basis_haiku_test.html 덮어씀)
"""
import os
import re
from collections import Counter

import numpy as np

from features import nearby

try:  # Haiku는 기본 꺼짐 → predict.py는 anthropic 패키지 없이도 동작
    import anthropic
except ImportError:
    anthropic = None

MODEL = "claude-haiku-5-5"
FEW_TRADES = 14  # 주변 거래 이 건수 이하면 신뢰도 이유로 표시 (6단계 표: 14건 이하 구간 적중률 44% 이하, AI가 정한 기준)
SIM_TOL, SIM_MIN = 0.2, 3  # 기준가 = 비슷한 면적(전용 ±20%) 거래, 이 건수 미만이면 면적 구분 없는 주변 거래 (10/13 사용자 결정)
PRICE = {"input": 0.10, "output": 0.50}  # Haiku 5.5 달러/백만 토큰 (공식 가격표, 2026-10-13 조회)
SYSTEM = """너는 빌라 시세 산출 근거를 한 줄로 다듬는 편집자다.
규칙:
1. 입력 문장의 숫자는 모두 쓰인 모양 그대로(+, - 부호, 소수점, 쉼표 포함) 한 번씩 남긴다. 숫자를 바꾸거나, 빼거나, 새로 만들지 않는다.
2. 입력에 없는 사실, 평가, 추측을 더하지 않는다.
3. 부동산을 잘 모르는 사람도 읽기 쉬운 자연스러운 한국어 한 문장(120자 안팎)으로 쓴다.
4. 다듬은 문장만 출력한다."""


def similar_area(key, lat, lon, ref, area, src):
    """비슷한 면적 기준가 재료 → (㎡당 중앙값, 건수). 전용면적 ±20% 거래만 남긴 목록으로 features.nearby를 행마다 다시 부름
    → 반경 500m·기준일 직전 12개월·같은 지번 제외 규칙이 모델의 주변 시세 변수와 똑같음. 좌표가 없으면 (빈칸, 빈칸)"""
    a = src["pool"].area_m2.astype(float).to_numpy()
    res = [nearby(key[i:i + 1], lat[i:i + 1], lon[i:i + 1], ref[i:i + 1],
                  {"pool": src["pool"][(a >= area[i] * (1 - SIM_TOL)) & (a <= area[i] * (1 + SIM_TOL))]}) for i in range(len(lat))]
    return np.array([p[0] for p, _ in res]), np.array([c[0] for _, c in res])


def template(f):
    """근거 숫자 → basis 문장. f: area_m2, est, sim_ppm, sim_cnt, nb_ppm_500m, nb_cnt_500m, pub_ppm, pub_ppm_missing, building_age, floor,
    station_dist_m, tier, spread_pct, area_note(선택: 면적을 입력 대신 어디서 가져왔는지)
    주어(추정가)와 기준가 계산식을 문장에 넣어 숫자마다 무엇을 가리키는지 분명히 함 (Haiku 시험에서 대상이 바뀐 오류 대응)
    기준가는 비슷한 면적 거래 우선 (작은 집은 ㎡당 가격이 높아 면적 구분 없이 비교하면 +157%처럼 크게 나옴, 10/13 사용자 결정)"""
    est, area = f["est"], f["area_m2"]
    if f["sim_cnt"] >= SIM_MIN:  # 빈칸(nan)이면 거짓 → 아래 면적 구분 없는 기준가
        ppm, near = f["sim_ppm"], f"반경 500m·최근 12개월 비슷한 면적(±20%) 다른 지번 {int(f['sim_cnt'])}건"
    else:
        ppm = f["nb_ppm_500m"]
        near = (f"비슷한 면적 거래가 적어 면적 구분 없이 반경 500m·최근 12개월 다른 지번 {int(f['nb_cnt_500m'])}건" if f["nb_cnt_500m"] > 0
                else "반경 500m 거래가 없어 넓힌 범위")
    base = ppm * area
    s = (f"추정 {est / 1e8:.2f}억: 주변 기준가 {base / 1e8:.2f}억({near}의 ㎡당 중앙값 {ppm / 1e4:,.0f}만원 × 전용 {area:.2f}㎡)"
         f"보다 {round((est / base - 1) * 100):+d}%")
    if f["pub_ppm_missing"]:
        s += ", 공시가격 정보 없음"
    else:
        pub = f["pub_ppm"] * area
        s += f", 공시가격 {pub / 1e8:.2f}억의 {est / pub:.2f}배"
    age, fl = int(round(f["building_age"])), int(f["floor"])
    s += (f". {f'준공 {age}년' if age >= 1 else '준공 1년 미만'}·{f'지하{-fl}층' if fl < 0 else f'{fl}층'}"  # 지하는 음수로 저장돼 있음
          f"·역 {int(round(f['station_dist_m'], -1)):,}m")
    if f["tier"] == "높음":
        s += ". 신뢰도 높음."
    else:  # 보통·낮음은 이유를 붙임: 세 모델 예측 차이는 항상 숫자로 (이유 없이 '낮음'만 나오던 사례 대응, AI가 정한 세부 규칙)
        why = (["주변 거래 적음"] if f["nb_cnt_500m"] <= FEW_TRADES else []) + [f"세 모델 예측 차이 {f['spread_pct']:.1f}%"]
        s += f". 신뢰도 {f['tier']}({', '.join(why)})."
    return s + (f" {f['area_note']}." if f.get("area_note") else "")


def numbers(s):
    """문장 속 숫자 묶음 (부호 포함, 쉼표 제거, 유니코드 빼기 → '-'). 순서는 무시하고 개수까지 비교"""
    return Counter(t.replace(",", "") for t in re.findall(r"[+-]?\d[\d,]*(?:\.\d+)?", s.replace("−", "-")))


def get_client():
    """ANTHROPIC_API_KEY가 있으면 클라이언트, 없으면 None (→ 템플릿만 사용)"""
    return anthropic.Anthropic(timeout=30.0, max_retries=2) if anthropic and os.environ.get("ANTHROPIC_API_KEY") else None


def polish(text, client):
    """Haiku 5.5로 표현만 다듬기 → (다듬은 문장 또는 None, 사유, 사용 토큰)"""
    r = client.messages.create(model=MODEL, max_tokens=1024, output_config={"effort": "low"}, system=SYSTEM,
                               messages=[{"role": "user", "content": text}])
    used = {"input": r.usage.input_tokens, "output": r.usage.output_tokens}
    out = " ".join(b.text for b in r.content if b.type == "text").strip()  # 생각 블록이 먼저 올 수 있어 종류로 골라 읽음
    if r.stop_reason != "end_turn" or not out:
        return None, f"응답 이상(stop_reason={r.stop_reason})", used
    return out.splitlines()[0].strip(), "", used


def make_basis(f, client=None):
    """→ (basis, 출처 'Haiku'·'템플릿', 템플릿을 쓴 사유, 사용 토큰)"""
    t = template(f)
    if client is None:
        return t, "템플릿", "LLM 사용 안 함", None
    try:
        out, why, used = polish(t, client)
    except anthropic.AuthenticationError:
        return t, "템플릿", "API 키 인증 실패", None
    except anthropic.RateLimitError:
        return t, "템플릿", "호출 한도 초과", None
    except anthropic.APIStatusError as e:
        return t, "템플릿", f"API 오류 {e.status_code}", None
    except anthropic.APIConnectionError:
        return t, "템플릿", "네트워크 오류·시간 초과", None
    if out is None:
        return t, "템플릿", why, used
    if numbers(out) != numbers(t):
        return t, "템플릿", "숫자 불일치", used
    return out, "Haiku", "", used


if __name__ == "__main__":
    import json
    from datetime import date
    from pathlib import Path

    import pandas as pd
    from dotenv import load_dotenv

    import sys

    from features import load_sources

    HAIKU = len(sys.argv) > 1 and sys.argv[1] == "haiku"  # Haiku 비교 재시험일 때만 API 호출
    ROOT = Path(__file__).resolve().parent.parent
    PROC, REPORT = ROOT / "data" / "processed", ROOT / "reports"
    load_dotenv(ROOT / ".env")

    # 숫자 검사 자체 시험: 순서만 바꾼 문장은 통과, 숫자 하나를 바꾸거나 부호를 뺀 문장은 실패해야 함
    t0 = "추정 5.51억: 주변 기준가 5.20억(반경 500m·최근 12개월 다른 지번 47건의 ㎡당 중앙값 1,250만원 × 전용 41.60㎡)보다 +6%, 공시가격 3.80억의 1.45배."
    assert numbers("공시가격 3.80억의 1.45배인 추정 5.51억은 반경 500m·최근 12개월 다른 지번 47건의 ㎡당 중앙값 1250만원 × 전용 41.60㎡ = 기준가 5.20억보다 +6%.") == numbers(t0)
    assert numbers(t0.replace("+6%", "+7%")) != numbers(t0)
    assert numbers(t0.replace("+6%", "6% 높음")) != numbers(t0)
    assert numbers(t0.replace("1.45배", "1.5배")) != numbers(t0)
    print("숫자 검사 자체 시험 통과 (순서 바뀜 → 통과, 숫자·부호 바뀜 → 걸러냄)")

    # 시험 대상: 학습용 최근 12개월 거래에서 권역별로 고루 10건 (+ 주변 거래 0건·공시가격 없음 사례 포함), 시험용은 안 씀
    P = json.loads((REPORT / "uncertainty_params.json").read_text(encoding="utf-8"))
    d = pd.read_csv(PROC / "oof_predictions.csv").merge(pd.read_csv(PROC / "preprocessed_all.csv").drop(columns="price_won"), on="id")
    d = d[d.deal_date >= "2025-10-01"].reset_index(drop=True)
    L = np.log(d[["pred_CatBoost", "pred_XGBoost", "pred_LightGBM"]].to_numpy())
    d["spread_pct"] = (np.exp(L.max(1) - L.min(1)) - 1) * 100
    X = np.column_stack([np.ones(len(d)), np.log1p(d.nb_cnt_500m), d.spread_pct, d.gu == "강남구", d.gu == "관악구"]).astype(float)
    d["conf"] = 1 / (1 + np.exp(-X @ np.array(P["confidence"]["계수"])))
    d["tier"] = np.array(P["등급"]["이름"])[np.digitize(d.conf, P["등급"]["경계"])]
    rng = np.random.default_rng(42)
    pick = [rng.choice(d.index[d.gu == g], n, replace=False) for g, n in [("강남구", 3), ("강서구", 3), ("관악구", 2)]]
    pick += [rng.choice(d.index[d.nb_cnt_500m == 0], 1), rng.choice(d.index[d.pub_ppm_missing == 1], 1)]
    s = d.loc[np.concatenate(pick)].rename(columns={"pred_final": "est"})
    # 비슷한 면적 기준가: 기준일 = 거래일 (모델의 주변 시세 변수와 같은 시점)
    src = load_sources()
    s = s.merge(pd.read_csv(PROC / "input_all.csv", dtype=str)[["id", "jibun"]].astype({"id": int}), on="id")
    geo = src["geo"].reindex(pd.MultiIndex.from_arrays([s.gu, s.dong, s.jibun]))
    s["sim_ppm"], s["sim_cnt"] = similar_area((s.gu + "|" + s.dong + "|" + s.jibun).to_numpy(), geo.lat.to_numpy(), geo.lon.to_numpy(),
                                              pd.to_datetime(s.deal_date).to_numpy(), s.area_m2.to_numpy(), src)

    client = get_client() if HAIKU else None
    print("모드:", ("Haiku 비교 재시험" + ("" if client else " — API 키 없음 → 템플릿만")) if HAIKU else "템플릿만 (API 호출 없음)")
    rows, cost = [], 0.0
    for _, f in s.iterrows():
        f = f.to_dict()
        b, src, why, used = make_basis(f, client)
        cost += sum(used[k] * PRICE[k] for k in used) / 1e6 if used else 0
        rows.append({"id": f["id"], "구": f["gu"], "실제(억)": round(f["price_won"] / 1e8, 2), "추정(억)": round(f["est"] / 1e8, 2),
                     "confidence": round(f["conf"], 2), "템플릿 문장": template(f), "최종 basis": b, "출처": src, "템플릿 사유": why or "—",
                     "토큰(입력/출력)": f"{used['input']}/{used['output']}" if used else "—"})
    out = pd.DataFrame(rows)
    n_llm = int((out.출처 == "Haiku").sum())
    if HAIKU:
        print(out[["id", "구", "출처", "템플릿 사유"]].to_string(index=False))
        print(f"Haiku 통과 {n_llm}/{len(out)} · 비용 ${cost:.5f}")
        file, title = "basis_haiku_test.html", f"Haiku 비교 재시험 — 모델 {MODEL} (effort low) · 숫자 검사 통과 {n_llm}/{len(out)} · 비용 ${cost:.5f}"
        rule = "<li>템플릿 문장 → Haiku가 표현만 다듬음 → 두 문장의 숫자 묶음이 같으면 Haiku 문장, 다르면 템플릿. 숫자 검사를 통과해도 숫자가 가리키는 대상이 바뀔 수 있으니 직접 읽어 확인</li>"
        table = out
    else:
        for b in out["템플릿 문장"]:
            print(" ", b)
        file, title = "basis_samples.html", "템플릿 (API 호출 없음) — predict.py가 쓰는 방식"
        rule = ("<li>숫자는 모두 모델·데이터 값 그대로. 주어(추정)와 기준가 계산식(㎡당 중앙값 × 전용면적)을 문장에 넣어 숫자마다 무엇을 가리키는지 분명히 함</li>"
                f"<li>주변 기준가 (사용자 결정): 반경 500m·최근 12개월·다른 지번 중 비슷한 면적(전용 ±{SIM_TOL:.0%}) 거래의 ㎡당 중앙값 × 면적. "
                f"{SIM_MIN}건 미만이면 면적 구분 없는 주변 거래</li>")
        table = out[["id", "구", "실제(억)", "추정(억)", "confidence", "최종 basis"]].rename(columns={"최종 basis": "basis"})

    css = (ROOT / "src" / "eda.py").read_text(encoding="utf-8").split("<style>")[1].split("</style>")[0].replace("{{", "{").replace("}}", "}")
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>basis 시험</title><style>{css}</style></head><body><div class="wrap">
<header><h1>basis 문장 시험 (학습용 최근 12개월 10건)</h1>
<p>{title} · 생성 {date.today()} · <code>python src/basis.py{' haiku' if HAIKU else ''}</code></p></header>
<section><h2>규칙</h2><ul>
{rule}
<li>신뢰도 이유 (AI가 정한 세부 규칙): 높음은 이유 없음. 보통·낮음은 세 모델 예측 차이(%)를 항상 표시하고, 주변 거래 {FEW_TRADES}건 이하면 "주변 거래 적음" 추가</li>
<li>시험 대상: 권역별 강남 3·강서 3·관악 2건 + 주변 거래 0건 1건 + 공시가격 없음 1건 (난수 42). 추정 = 교차검증 예측(CatBoost). 시험용 데이터는 안 씀</li></ul></section>
<section><h2>결과</h2>{'<div class="tw">' + table.to_html(index=False, border=0, escape=True) + '</div>'}</section>
</div></body></html>"""
    (REPORT / file).write_text(html, encoding="utf-8")
    print(f"저장: reports/{file}")
