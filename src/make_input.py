"""실거래 원본 → 과제 입력 규격(input.csv) 형식 + 정답 파일

대상: 화곡동·강남구·관악구 '다세대', 해제 거래 제외 (연립 제외 근거는 결정기록.md)
저장:
  data/processed/input_all.csv  — id, sigungu, dong, jibun, floor, ho, area_m2 (과제 규격 그대로)
  data/processed/answer_all.csv — id, price_won, deal_date (실제 거래가: 만원 → 원)
ho: 실거래 원본에 호가 없음 → 2026 공시가격 목록에서 '같은 지번 + 같은 층 + 같은 전용면적'인 호가
    딱 1개일 때만 그 호로 채운 추정값 (10/11 결정, 근거는 결정기록.md 2번)
    후보가 0개·2개 이상이거나, 거래일 이후 사용승인된 건물(재건축 의심)이면 빈칸
실행: python src/make_input.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "processed"

df = pd.read_csv(ROOT / "data" / "raw" / "trades_rh.csv", dtype=str)
df = df[((df.gu != "강서구") | (df.umdNm == "화곡동"))
        & (df.houseType == "다세대")
        & (df.cdealType.fillna("").str.strip() != "O")].reset_index(drop=True)
df["id"] = range(1, len(df) + 1)
df["deal_date"] = pd.to_datetime(df.dealYear + "-" + df.dealMonth + "-" + df.dealDay)

# 공시가격: API가 같은 호를 2번씩 돌려줘서 중복 제거. 면적은 소수 2자리로 정확히 비교
# (1자리로 반올림하면 23.52㎡·23.49㎡처럼 다른 호가 같은 값이 돼 후보가 늘고, 틀린 호를 고를 위험도 커짐)
price = pd.read_csv(ROOT / "data" / "raw" / "house_price_2026.csv", dtype=str)
price = price[price["_error"].isna()].drop_duplicates()
price["ar"] = price.prvuseAr.astype(float).round(2)
key = ["gu", "umdNm", "jibun", "floor", "ar"]
cand = price.rename(columns={"floorNm": "floor"}).groupby(key).hoNm.agg(["count", "first"]).reset_index()
df["ar"] = df.excluUseAr.astype(float).round(2)
df = df.merge(cand, on=key, how="left")
df["ho"] = df["first"].where(df["count"] == 1, "")

# 거래 후에 사용승인된 건물 = 거래 당시와 다른 건물일 수 있음 → 2026 공시가격의 호를 쓰면 안 됨
title = pd.read_csv(ROOT / "data" / "raw" / "building_title.csv", dtype=str)
apr = pd.to_datetime(title.useAprDay, format="%Y%m%d", errors="coerce").groupby([title.gu, title.umdNm, title.jibun]).max()
df["apr"] = [apr.get(k) for k in zip(df.gu, df.umdNm, df.jibun)]
rebuilt = df.apr > df.deal_date
df.loc[rebuilt, "ho"] = ""

inp = pd.DataFrame({
    "id": df.id,
    "sigungu": "서울특별시 " + df.gu,
    "dong": df.umdNm,
    "jibun": df.jibun,
    "floor": df.floor,
    "ho": df.ho,
    "area_m2": df.excluUseAr,
})
ans = pd.DataFrame({
    "id": df.id,
    "price_won": df.dealAmount.str.replace(",", "").astype(int) * 10_000,
    "deal_date": df.deal_date.dt.strftime("%Y-%m-%d"),
})

OUT_DIR.mkdir(parents=True, exist_ok=True)
inp.to_csv(OUT_DIR / "input_all.csv", index=False, encoding="utf-8-sig")
ans.to_csv(OUT_DIR / "answer_all.csv", index=False, encoding="utf-8-sig")
print(f"저장: input_all.csv / answer_all.csv ({len(inp):,}행)")
print(f"ho 채움 {(inp.ho != '').sum():,}건 | 후보 0개 {df['count'].isna().sum():,} · 2개 이상 {(df['count'] > 1).sum():,}"
      f" · 재건축 의심으로 비움 {(rebuilt & (df['count'] == 1)).sum():,}")
