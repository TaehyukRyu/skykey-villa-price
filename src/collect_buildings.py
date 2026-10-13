"""건축물대장 표제부 + 공동주택 공시가격 수집 (다세대 실거래 지번 기준)

- 건축물대장 표제부 (data.go.kr 건축HUB 15134735 getBrTitleInfo): 사용승인일, 층수, 승강기, 세대수, 구조, 주차 등
  → data/raw/building_title.csv
- 공동주택 공시가격 2026년 (VWorld 국가중점 getApartHousingPriceAttr): 동·호별 공시가격, 전용면적, 층
  → data/raw/house_price_2026.csv

지번 → 코드: data/raw/geocode_jibun.csv의 PNU 앞 10자리(법정동코드)를 쓰고, 번·지는 실거래 지번에서 직접 만든다.
(지오코더가 근처 다른 필지로 대체 매칭한 경우에도 원래 지번을 조회하기 위함)
전유부(호별 면적)는 학습에 필요 없어서(실거래에 전용면적 있음) 일괄 수집하지 않고, predict.py에서 면적이 빈 물건만 실행 중 조회한다.
실행: python src/collect_buildings.py            (둘 다)
      python src/collect_buildings.py price      (공시가격만, 2026년)
      python src/collect_buildings.py price 2024 (공시가격, 연도 지정 → house_price_2024.csv)
과거 연도는 시점 누수 방지용: 거래마다 거래일 당시 이미 공시돼 있던 연도의 값을 붙이기 위함 (결정기록.md 7번)
중간에 멈춰도 다시 실행하면 이어서 받는다.
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
load_dotenv(ROOT / ".env")
DATA_KEY = unquote(os.environ.get("DATA_GO_KR_KEY", ""))  # 키가 없어도 import는 됨 (predict.py가 수집된 지번만 쓸 때). 호출하면 오류 응답
VWORLD_KEY = os.environ.get("VWORLD_KEY", "")
TITLE_URL = "https://apis.data.go.kr/1613000/BldRgstHubService/getBrTitleInfo"
PRICE_URL = "https://api.vworld.kr/ned/data/getApartHousingPriceAttr"


def targets():
    g = pd.read_csv(RAW / "geocode_jibun.csv", dtype=str)
    ok = g[g.status == "OK"]
    dong_code = ok.assign(code=ok.pnu.str[:10]).groupby(["gu", "umdNm"]).code.agg(lambda s: s.mode()[0])
    g["dong_code"] = [dong_code.get((gu, d)) for gu, d in zip(g.gu, g.umdNm)]
    bun_ji = g.jibun.str.split("-", expand=True).reindex(columns=[0, 1]).fillna("0")
    g["bun"], g["ji"] = bun_ji[0].str.zfill(4), bun_ji[1].str.zfill(4)
    g["pnu_jibun"] = g.dong_code + "1" + g.bun + g.ji  # 1 = 일반 필지(산 아님)
    return g[g.dong_code.notna()][["gu", "umdNm", "jibun", "dong_code", "bun", "ji", "pnu_jibun"]]


def get_json(url, params):
    for attempt in range(4):
        try:
            return requests.get(url, params=params, timeout=30).json()
        except Exception:  # 호출이 몰려 차단되면 JSON 대신 오류 페이지가 옴 → 쉬었다가 재시도
            time.sleep(2 ** attempt)
    return None


def fetch_title(t):
    rows, page = [], 1
    while True:
        j = get_json(TITLE_URL, dict(serviceKey=DATA_KEY, sigunguCd=t["dong_code"][:5], bjdongCd=t["dong_code"][5:],
                                     platGbCd="0", bun=t["bun"], ji=t["ji"], numOfRows="100", pageNo=str(page), _type="json"))
        if j is None or "response" not in j:
            return [dict(gu=t["gu"], umdNm=t["umdNm"], jibun=t["jibun"], _error="request_failed")]
        body = j["response"]["body"]
        items = (body.get("items") or {}).get("item") or []
        items = items if isinstance(items, list) else [items]
        rows += [dict(gu=t["gu"], umdNm=t["umdNm"], jibun=t["jibun"], **it) for it in items]
        if len(rows) >= int(body.get("totalCount") or 0) or not items:
            return rows or [dict(gu=t["gu"], umdNm=t["umdNm"], jibun=t["jibun"], _error="no_record")]
        page += 1


def fetch_price(t, year="2026"):
    rows, page = [], 1
    while True:
        j = get_json(PRICE_URL, dict(pnu=t["pnu_jibun"], stdrYear=year, format="json", numOfRows="1000",
                                     pageNo=str(page), key=VWORLD_KEY,
                                     domain="http://localhost"))  # 키 발급 때 적은 서비스URL. 없으면 INCORRECT_KEY
        if j is None:
            return [dict(gu=t["gu"], umdNm=t["umdNm"], jibun=t["jibun"], _error="request_failed")]
        top = next(iter(j.values()))
        if top.get("resultCode"):  # 정상이면 빈 값, 오류면 코드가 들어옴 (예: 호출 한도 초과)
            return [dict(gu=t["gu"], umdNm=t["umdNm"], jibun=t["jibun"], _error=top["resultCode"])]
        items = top.get("field") or []
        rows += [dict(gu=t["gu"], umdNm=t["umdNm"], jibun=t["jibun"], **it) for it in items]
        if len(rows) >= int(top.get("totalCount") or 0) or not items:
            return rows or [dict(gu=t["gu"], umdNm=t["umdNm"], jibun=t["jibun"], _error="no_record")]
        page += 1


def collect(fn, tlist, out_name, chunk=500):
    out = RAW / out_name
    rows = []
    if out.exists():  # 이전 실행에서 성공한 지번은 건너뛰고, 실패·미완료 지번만 조회
        done = pd.read_csv(out, dtype=str)
        if "_error" in done:
            done = done[done["_error"].isna()]
        rows = done.to_dict("records")
        keys = set(zip(done.gu, done.umdNm, done.jibun))
        tlist = [t for t in tlist if (t["gu"], t["umdNm"], t["jibun"]) not in keys]
    print(f"{out_name}: 조회할 지번 {len(tlist):,}개", flush=True)
    for i in range(0, len(tlist), chunk):
        with ThreadPoolExecutor(max_workers=2) as pool:
            rows += [r for rs in pool.map(fn, tlist[i:i + chunk]) for r in rs]
        pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")  # 500지번마다 저장 → 중간에 멈춰도 그때까지는 남음
        print(f"  {min(i + chunk, len(tlist)):,}/{len(tlist):,}", flush=True)
    df = pd.DataFrame(rows)
    err = df["_error"].value_counts().to_dict() if "_error" in df else {}
    print(f"저장: {out_name} ({len(df):,}행) | 오류/없음: {err}")


def main():
    parts = sys.argv[1:] or ["title", "price"]  # 예: python src/collect_buildings.py price 2024
    year = next((p for p in parts if p.isdigit()), "2026")
    tlist = targets().to_dict("records")
    print(f"대상 지번 {len(tlist):,}개")
    if "title" in parts:
        collect(fetch_title, tlist, "building_title.csv")
    if "price" in parts:
        collect(partial(fetch_price, year=year), tlist, f"house_price_{year}.csv")


if __name__ == "__main__":
    main()
