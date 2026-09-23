import re
import time
import random
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = "https://kato.kr"
HEADERS = {"User-Agent": "CourtVisionStudentProject/0.1 (course assignment; +https://github.com/Jenny5789/CourtVision)"}
ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = ROOT / "data" / "crawl"
DATE_RE = re.compile(r"(\d{4})\.(\d{2})\.(\d{2})\s*~\s*(\d{4})\.(\d{2})\.(\d{2})")
ID_RE = re.compile(r"/openGame/(\d+)")
GROUP_RE = re.compile(r"group([a-z0-9]+)_\d+\.png", re.I)


def polite_sleep(base=1.5, jitter=0.5):
    time.sleep(base + random.uniform(0, jitter))


def fetch_year_html(year, use_cache=True, retries=3):
    """해당 연도 목록 HTML. 아직 없는 연도(404)면 None"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"kato_{year}.html"
    if use_cache and cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            polite_sleep()
            r = requests.get(f"{BASE}/openList/{year}", headers=HEADERS, timeout=15)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            r.encoding = r.apparent_encoding
            cache_path.write_text(r.text, encoding="utf-8")
            return r.text
        except requests.RequestException as e:
            last_err = e
            print(f"  요청 실패({attempt}/{retries}): {e}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"KATO {year}년 수집 실패: {last_err}")


def parse_html(html):
    soup = BeautifulSoup(html, "html.parser")
    cards = [t for t in soup.find_all("table") if t.find("td", class_="title-sector")]

    rows = []
    for t in cards:
        a = t.select_one(".title a.content-title")
        date_el = t.select_one(".date")
        if not a or not date_el:
            continue
        m = DATE_RE.search(date_el.get_text())
        if not m:
            continue
        y1, m1, d1, y2, m2, d2 = m.groups()

        href = a.get("href", "")
        idm = ID_RE.search(href)
        parts_el = t.select_one(".parts")
        divisions = [p.strip() for p in parts_el.get_text().split(",")] if parts_el else []
        registration = "/".join(dict.fromkeys(s.get_text(strip=True) for s in t.select(".part-sector span")))
        img = t.select_one(".group-sector img")
        gm = GROUP_RE.search(img.get("src", "")) if img else None

        rows.append({
            "source": "KATO",
            "source_id": idm.group(1) if idm else "",
            "name": " ".join(a.get_text().split()),
            "start_date": f"{y1}-{m1}-{d1}",
            "end_date": f"{y2}-{m2}-{d2}",
            "divisions": ",".join(divisions),
            "registration": registration,
            "ranking_group": gm.group(1).upper() if gm else "",
            "url": BASE + href if href.startswith("/") else href,
        })
    return rows


if __name__ == "__main__":
    this_year = date.today().year
    all_rows = []
    for year in [this_year, this_year + 1]:
        print(f"[{year}] 수집 중...")
        html = fetch_year_html(year)
        if html is None:
            print("  아직 공개되지 않은 연도(404)")
            continue
        rows = parse_html(html)
        if not rows and year == this_year:
            raise ValueError("올해 대회를 한 건도 못 찾음 - 사이트 구조가 바뀌었을 수 있음")
        print(f"  {len(rows)}건")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    print(f"\n총 {len(df)}건 (ID 중복 {df.duplicated(subset=['source_id']).sum()}건)")
    print("기간:", df["start_date"].min(), "~", df["end_date"].max())
    print("접수 상태별:", df["registration"].value_counts().to_dict())
    print("랭킹 그룹별:", df["ranking_group"].value_counts().to_dict())
    print()
    for r in all_rows[:3]:
        print(r["start_date"], "|", r["name"], "|", r["divisions"], "|", r["registration"], "|", r["url"])

    out = CACHE_DIR / "kato_tournaments.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n저장: {out}")