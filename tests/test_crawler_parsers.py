from datetime import datetime, timedelta

import pytest

from crawler import kato, kta, run_crawl, store

KATO_HTML = """
<html><body>
<table><tr><td class="title-sector">
  <div class="title"><a class="content-title" href="/openGame/123">제1회   테스트
     오픈</a></div>
  <div class="date">2026.05.01 ~ 2026.05.03</div>
  <div class="parts">남자부, 여자부</div>
  <div class="part-sector"><span>접수중</span><span>접수중</span></div>
  <div class="group-sector"><img src="/img/groupa_1.png"></div>
</td></tr></table>
<table><tr><td class="title-sector">
  <div class="title"><a class="content-title" href="/openGame/124">날짜 없는 대회</a></div>
  <div class="date">일정 미정</div>
</td></tr></table>
<table><tr><td>관련 없는 표</td></tr></table>
</body></html>
"""


def kta_table(rows):
    body = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows
    )
    return ("<table><tr><th>번호</th><th>대회기간</th><th>기간</th><th>대회명칭</th><th>장소</th></tr>"
            + body + "</table>")


def kta_html(tables=3):
    sections = [
        [["1", "2026.03.01 ~ 2026.03.03", "3일", "예산 대회", "서울"],
         ["2", "날짜형식아님", "-", "건너뛸 대회", "부산"],
         ["x", "2026.03.05 ~ 2026.03.06", "2일", "칸 수가 모자란 행"]],
        [["1", "2026.04.10 ~ 2026.04.12", "3일", "주관 대회", "대구"]],
        [],
    ][:tables]
    return "<html><body>" + "".join(kta_table(r) for r in sections) + "</body></html>"


# ---- KATO ----
def test_kato_parse_extracts_fields_and_normalizes_text():
    rows = kato.parse_html(KATO_HTML)
    assert len(rows) == 1
    r = rows[0]
    assert r["source"] == "KATO" and r["source_id"] == "123"
    assert r["name"] == "제1회 테스트 오픈"
    assert (r["start_date"], r["end_date"]) == ("2026-05-01", "2026-05-03")
    assert r["divisions"] == "남자부,여자부"
    assert r["registration"] == "접수중"
    assert r["ranking_group"] == "A"
    assert r["url"] == "https://kato.kr/openGame/123"


def test_kato_parse_empty_html_returns_no_rows():
    assert kato.parse_html("<html></html>") == []


# ---- KTA ----
def test_kta_parse_extracts_rows_per_section_and_skips_bad_rows():
    rows = kta.parse_html(kta_html(), 2026)
    assert [(r["section"], r["name"]) for r in rows] == [("예산사업", "예산 대회"), ("주관사업", "주관 대회")]
    assert rows[0]["start_date"] == "2026-03-01" and rows[0]["end_date"] == "2026-03-03"
    assert rows[0]["venue"] == "서울" and rows[0]["year"] == 2026


def test_kta_parse_raises_when_table_count_changes():
    with pytest.raises(ValueError, match="표 개수"):
        kta.parse_html(kta_html(tables=2), 2026)


# ---- 수집 + 저장 연결 ----
def test_refresh_if_stale_skips_recent_and_collects_when_old(db, monkeypatch):
    calls = []
    monkeypatch.setattr(run_crawl, "collect", lambda refresh=False: calls.append(refresh) or [{
        "source": "KATO", "source_id": "1", "name": "n", "start_date": "2099-01-01", "end_date": "2099-01-02",
    }])

    assert run_crawl.refresh_if_stale(max_age_hours=24) == {"new": 1, "changed": 0, "unchanged": 0}
    assert calls == [True]

    assert run_crawl.refresh_if_stale(max_age_hours=24) is None
    assert calls == [True]

    old = (datetime.now() - timedelta(hours=30)).isoformat(timespec="seconds")
    con = store.connect()
    con.execute("UPDATE tournaments SET last_seen=?", (old,))
    con.commit()
    con.close()
    stats = run_crawl.refresh_if_stale(max_age_hours=24)
    assert stats["unchanged"] == 1 and len(calls) == 2
