"""서울 지하철역 좌표 수집 (서울 열린데이터광장 OA-21232 역사마스터, 전 노선)

저장: data/raw/subway_stations.csv
실행: python src/collect_stations.py
"""
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
URL = f"http://openapi.seoul.go.kr:8088/{os.environ['SEOUL_KEY']}/json/subwayStationMaster/1/1000/"
OUT = ROOT / "data" / "raw" / "subway_stations.csv"

data = requests.get(URL, timeout=30).json()["subwayStationMaster"]
df = pd.DataFrame(data["row"])
assert len(df) == data["list_total_count"], "한 번에 다 못 받음 — 페이지 나눠 받기 필요"
df.to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"저장: {OUT} ({len(df)}행) | 노선 수: {df.ROUTE.nunique()}")
