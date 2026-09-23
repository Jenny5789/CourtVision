import os
import requests
from bs4 import BeautifulSoup

URL = "https://www.kortennis.or.kr/cmpt/92/cmptList.do"
HEADERS = {"User-Agent": "CourtVisionStudentProject/0.1 (course assignment; +https://github.com/Jenny5789/CourtVision)"}

os.makedirs("data/crawl", exist_ok=True)

r = requests.get(URL, headers=HEADERS, timeout=15)
print("status:", r.status_code, "| content-type:", r.headers.get("Content-Type"), "| bytes:", len(r.content))

r.encoding = r.apparent_encoding
with open("data/crawl/kta_raw.html", "w", encoding="utf-8") as f:
    f.write(r.text)

soup = BeautifulSoup(r.text, "html.parser")

tables = soup.find_all("table")
print("table 개수:", len(tables))
for ti, t in enumerate(tables):
    headers = [th.get_text(strip=True) for th in t.find_all("th")]
    rows = t.find_all("tr")
    print(f"\n[table {ti}] 헤더: {headers} | 행 수: {len(rows)}")
    for tr in rows[1:4]:
        print("  ", [td.get_text(" ", strip=True) for td in tr.find_all("td")])

print("\n[select 요소 (연도 필터 후보)]")
for s in soup.find_all("select"):
    opts = [o.get("value") for o in s.find_all("option")][:5]
    print("  name =", s.get("name"), "| id =", s.get("id"), "| 옵션 예시:", opts)

print("\n[form 요소]")
for f in soup.find_all("form"):
    print("  action =", f.get("action"), "| method =", f.get("method"))