import os
import re
import requests
from bs4 import BeautifulSoup

URL = "https://kato.kr/openList"
HEADERS = {"User-Agent": "CourtVisionStudentProject/0.1 (course assignment; +https://github.com/Jenny5789/CourtVision)"}

os.makedirs("data/crawl", exist_ok=True)

r = requests.get(URL, headers=HEADERS, timeout=15)
print("status:", r.status_code, "| content-type:", r.headers.get("Content-Type"), "| bytes:", len(r.content))

r.encoding = r.apparent_encoding
with open("data/crawl/kato_raw.html", "w", encoding="utf-8") as f:
    f.write(r.text)

html = r.text
soup = BeautifulSoup(html, "html.parser")

print("\n[섹션 제목이 HTML에 있는가]")
for word in ["경기중인 대회", "접수중인 대회", "접수예정 대회"]:
    print(f"  {word}: {html.count(word)}번")

print("\n[대회 이름이 HTML 안에 실제로 들어 있는가 (서버 렌더링 여부)]")
for word in ["동호인테니스대회", "전국", "개나리부", "접수"]:
    print(f"  '{word}': {html.count(word)}번")

print("\n[구조 요약]")
print("  table:", len(soup.find_all("table")), "| li:", len(soup.find_all("li")), "| a:", len(soup.find_all("a")))
hrefs = [a.get("href") for a in soup.find_all("a") if a.get("href") and ("open" in a.get("href") or "view" in a.get("href").lower())]
print("  대회 관련 링크 예시:", hrefs[:8])

print("\n[script 파일]")
for s in soup.find_all("script"):
    if s.get("src"):
        print("  ", s.get("src"))

print("\n[페이지 안에서 API/AJAX 호출로 보이는 부분]")
pattern = re.compile(r"(fetch\(|axios|\$\.ajax|\$\.get|\$\.post|XMLHttpRequest|/api/|\.json|url\s*:)", re.I)
count = 0
for line in html.splitlines():
    if pattern.search(line):
        print("  ", line.strip()[:200])
        count += 1
        if count >= 15:
            break
if count == 0:
    print("  (없음)")