"""확인용: prep.py가 모델 학습 때 메모리에서 만드는 입력 데이터를 파일로 저장 (사람이 눈으로 보는 용도)

모델은 이 파일을 쓰지 않고 preprocessed_all.csv + prep.py를 직접 씀 → prep.py를 바꾸면 이 스크립트를 다시 실행
기준(범주 목록 등)은 학습용 행으로만 맞추고(fit), 학습·시험 행을 모두 변환
저장: data/processed/model_input_regression.csv (회귀용 31칸), model_input_tree.csv (트리용 16칸)
      + 앞뒤 확인용 칸: id, set(train/test), price_won(원 단위 정답), target_log(학습에 쓰는 log 정답)
실행: python src/export_model_input.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from prep import TARGET, regression_prep, tree_prep

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"

df = pd.read_csv(PROC / "preprocessed_all.csv", float_precision="round_trip").merge(pd.read_csv(PROC / "split.csv"), on="id")
train = df[df.set == "train"]
for name, prep in [("regression", regression_prep()), ("tree", tree_prep())]:
    prep.fit(train)
    X = pd.DataFrame(prep.transform(df), columns=prep.get_feature_names_out())
    X = X.apply(lambda c: c.astype(int) if c.isin([0, 1]).all() else c)  # 원핫·표시 칸은 0/1 정수로 (읽기 쉽게)
    out = pd.concat([df[["id", "set"]].reset_index(drop=True), X,
                     df[[TARGET]].reset_index(drop=True).assign(target_log=np.log(df[TARGET].to_numpy()))], axis=1)
    out.to_csv(PROC / f"model_input_{name}.csv", index=False, encoding="utf-8-sig")
    print(f"저장: model_input_{name}.csv ({len(out):,}행 × 변환 칸 {X.shape[1]} + 확인용 칸 4)")
