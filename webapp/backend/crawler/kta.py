import re
import time
import random
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

URL = "https://www.kortennis.or.kr/cmpt/92/cmptList.do"
HEADERS = {"User-Agent": "CourtVisionStudentProject/0.1 (course assignment; +https://github.com/Jenny5789/CourtVision)"}
ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = ROOT / "data" / "crawl"
SECTIONS = ["예산사업", "주관사업", "비예산사업"]
EXPECTED_HEADER = ["번호", "대회기간", "기간", "대회명칭", "장소"]
DATE_RE = re.compile(r"(\d{4})\.(\d{2})\.(\d{2})\s*~\s*(\d{4})\.(\d{2})\.(\d{2})")


def polite_sleep(base=1.5, jitter=0.5):
    time.sleep(base + random.uniform(0, jitter))


def fetch_year_html(year, use_cache=True, retries=3):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"kta_{year}.html"
    if use_cache and cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            polite_sleep()
            r = requests.post(
                URL, headers=HEADERS, timeout=15,
                data={"pageIndex": "1", "businessCd": "", "cmptYear": str(year), "cmptCd": "92"},
            )
            r.raise_for_status()
            r.encoding = r.apparent_encoding
            cache_path.write_text(r.text, encoding="utf-8")
            return r.text
        except requests.RequestException as e:
            last_err = e
            print(f"  요청 실패({attempt}/{retries}): {e}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"{year}년 수집 실패: {last_err}")


def parse_html(html, year):
    soup = BeautifulSoup(html, "html.parser")
    tables = [
        t for t in soup.find_all("table")
        if [th.get_text(strip=True) for th in t.find_all("th")] == EXPECTED_HEADER
    ]
    if len(tables) != len(SECTIONS):
        raise ValueError(f"표 개수 이상: {len(tables)}개 (기대 {len(SECTIONS)}개) - 사이트 구조가 바뀌었을 수 있음")

    rows = []
    for section, table in zip(SECTIONS, tables):
        for tr in table.find_all("tr")[1:]:
            tds = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
            if len(tds) != 5:
                continue
            m = DATE_RE.search(tds[1])
            if not m:
                continue
            y1, m1, d1, y2, m2, d2 = m.groups()
            rows.append({
                "source": "KTA",
                "year": year,
                "section": section,
                "name": tds[3],
                "start_date": f"{y1}-{m1}-{d1}",
                "end_date": f"{y2}-{m2}-{d2}",
                "venue": tds[4],
            })
    return rows


if __name__ == "__main__":
    all_rows = []
    for year in [2025, 2026]:
        print(f"[{year}] 수집 중...")
        html = fetch_year_html(year)
        rows = parse_html(html, year)

        wrong_year = [r for r in rows if not r["start_date"].startswith(str(year))]
        if rows and len(wrong_year) == len(rows):
            print(f"  경고: {year}년을 요청했는데 다른 연도 데이터가 왔습니다. POST 파라미터를 다시 확인하세요.")

        print(f"  {len(rows)}건 (섹션별: " + ", ".join(
            f"{s} {sum(1 for r in rows if r['section'] == s)}" for s in SECTIONS) + ")")
        for r in rows[:2]:
            print("   ", r["start_date"], r["name"], "|", r["venue"])
        all_rows.extend(rows)

    out = CACHE_DIR / "kta_tournaments.csv"
    pd.DataFrame(all_rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n저장: {out} (총 {len(all_rows)}건)")