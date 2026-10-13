"""다세대 실거래 지번 → 좌표·PNU 변환 (VWorld 지오코더)

대상: data/raw/trades_rh.csv 중 화곡동·강남구·관악구 '다세대' 지번 (중복 제거)
저장: data/raw/geocode_jibun.csv
  - pnu: 필지 고유번호 19자리 (건축물대장·공시가격 조회 열쇠)
  - match: VWorld가 돌려준 지번이 입력 지번과 같은지 (다르면 근처 다른 필지로 대체된 것)
실행: python src/collect_geocode.py
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
KEY = os.environ.get("VWORLD_KEY", "")  # 키가 없어도 import는 됨 (predict.py가 수집된 지번만 쓸 때)
URL = "https://api.vworld.kr/req/address"
OUT = ROOT / "data" / "raw" / "geocode_jibun.csv"


def geocode(gu, dong, jibun):
    addr = f"서울특별시 {gu} {dong} {jibun}"
    row = dict(gu=gu, umdNm=dong, jibun=jibun, status="ERROR", matched="", match=False, pnu="", lon="", lat="", error="")
    for attempt in range(4):
        try:
            res = requests.get(URL, params=dict(service="address", request="getcoord", version="2.0", crs="epsg:4326",
                                                type="parcel", address=addr, format="json", key=KEY), timeout=20).json()["response"]
            if res.get("status") != "ERROR":
                break
            row["error"] = str(res.get("error", {}).get("text", ""))[:100]
        except Exception as e:
            row["error"] = type(e).__name__
        time.sleep(2 ** attempt)  # 1, 2, 4, 8초 쉬었다가 재시도
    else:
        return row
    row["status"], row["error"] = res.get("status", ""), ""
    if row["status"] == "OK":
        st = res["refined"]["structure"]
        row.update(matched=res["refined"]["text"], match=(st.get("level5") == jibun), pnu=st.get("level4LC", ""),
                   lon=res["result"]["point"]["x"], lat=res["result"]["point"]["y"])
    return row


def main():
    df = pd.read_csv(ROOT / "data" / "raw" / "trades_rh.csv", dtype=str)
    df = df[((df.gu != "강서구") | (df.umdNm == "화곡동")) & (df.houseType == "다세대")]
    df = df[df.jibun.fillna("").str.fullmatch(r"\d+(-\d+)?")]
    targets = df[["gu", "umdNm", "jibun"]].drop_duplicates().values.tolist()
    done = pd.DataFrame()
    if OUT.exists():  # 이전 실행에서 성공한 지번은 건너뛰고 실패한 것만 다시 조회
        prev = pd.read_csv(OUT, dtype=str)
        done = prev[prev.status.isin(["OK", "NOT_FOUND"])]
        ok_keys = set(map(tuple, done[["gu", "umdNm", "jibun"]].values.tolist()))
        targets = [t for t in targets if tuple(t) not in ok_keys]
    print(f"조회할 지번 {len(targets):,}개 (이미 완료 {len(done):,}개)")
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(lambda t: geocode(*t), targets))
    out = pd.concat([done, pd.DataFrame(rows).astype(str)], ignore_index=True)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"저장: {OUT} ({len(out):,}행)")
    print("status:", out.status.value_counts().to_dict(), "| 지번 일치:", int((out.match.astype(str) == "True").sum()))
    if "error" in out:
        print("오류 사유:", out.loc[out.status == "ERROR", "error"].value_counts().head(5).to_dict())


if __name__ == "__main__":
    main()
