"""연립다세대 매매 실거래 수집 (국토교통부 API, data.go.kr 15126467)

대상: 강서구(화곡동 포함)·강남구·관악구, 2023-10 ~ 2026-10 (37개월)
저장: data/raw/trades_rh.csv  — API 응답 칸을 손대지 않고 그대로 저장 (정제는 전처리 단계에서)
실행: python src/collect_trades.py
"""
import time
from pathlib import Path
from urllib.parse import unquote
import os

import pandas as pd
import requests
from defusedxml import ElementTree as ET
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
# 포털의 Encoding 키를 넣어도 동작하도록 한 번 풀어 둔다 (requests가 다시 인코딩함)
KEY = unquote(os.environ["DATA_GO_KR_KEY"])

URL = "https://apis.data.go.kr/1613000/RTMSDataSvcRHTrade/getRTMSDataSvcRHTrade"
REGIONS = {"11500": "강서구", "11680": "강남구", "11620": "관악구"}  # API는 구 단위 → 화곡동은 나중에 umdNm으로 거름
MONTHS = pd.period_range("2023-10", "2026-10", freq="M").strftime("%Y%m").tolist()
OUT = ROOT / "data" / "raw" / "trades_rh.csv"


def fetch_month(lawd_cd, ymd):
    rows, page = [], 1
    while True:
        for attempt in range(3):
            try:
                r = requests.get(URL, params=dict(serviceKey=KEY, LAWD_CD=lawd_cd, DEAL_YMD=ymd,
                                                  numOfRows="1000", pageNo=str(page)), timeout=30)
                root = ET.fromstring(r.content)
                break
            except Exception as e:
                if attempt == 2:
                    raise RuntimeError(f"{lawd_cd} {ymd} p{page} 실패: {type(e).__name__}")
                time.sleep(2)
        code = root.findtext(".//resultCode")
        if code != "000":
            raise RuntimeError(f"{lawd_cd} {ymd}: {code} {root.findtext('.//resultMsg')} {root.findtext('.//errMsg')}")
        items = [{c.tag: (c.text or "").strip() for c in it} for it in root.findall(".//item")]
        rows += items
        total = int(root.findtext(".//totalCount") or 0)
        if len(rows) >= total or not items:
            return rows
        page += 1


def main():
    all_rows = []
    for lawd_cd, gu in REGIONS.items():
        for ymd in MONTHS:
            rows = fetch_month(lawd_cd, ymd)
            for row in rows:
                row["LAWD_CD"], row["gu"], row["DEAL_YMD"] = lawd_cd, gu, ymd
            all_rows += rows
            time.sleep(0.1)
        print(f"{gu}: 누적 {len(all_rows):,}건")
    df = pd.DataFrame(all_rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"저장: {OUT} ({len(df):,}행, {df.shape[1]}칸)")


if __name__ == "__main__":
    main()
