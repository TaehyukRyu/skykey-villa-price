"""시험용 지번 떼어두기 (EDA·학습 전에 한 번만 실행)

- 지번 단위로 나눔: 같은 건물이 학습과 시험에 동시에 들어가면 모델이 그 건물 가격을 외워서 맞히므로
- 권역(구)마다 지번의 20%를 무작위로 시험용으로 (권역 비율이 학습·시험에서 같도록)
- 시드 고정(42) → 다시 실행해도 같은 결과
저장: data/processed/split.csv (id, set = train / test)
실행: python src/split.py
"""
from pathlib import Path

import pandas as pd

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"

inp = pd.read_csv(PROC / "input_all.csv", dtype=str)
jibun = inp[["sigungu", "dong", "jibun"]].drop_duplicates()
test = jibun.groupby("sigungu").sample(frac=0.2, random_state=42)
inp = inp.merge(test.assign(set="test"), on=["sigungu", "dong", "jibun"], how="left")
inp["set"] = inp["set"].fillna("train")
inp[["id", "set"]].to_csv(PROC / "split.csv", index=False, encoding="utf-8-sig")
print(inp.groupby(["sigungu", "set"]).size().unstack().assign(시험비율=lambda t: (t.test / (t.test + t.train)).round(3)))
