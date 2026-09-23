import sys
from collections import Counter
from datetime import date, datetime

try:                                    # 서버에서 패키지로 불러올 때
    from . import kta, kato, store
except ImportError:                     # python run_crawl.py 로 직접 실행할 때
    import kta
    import kato
    import store

KTA_LIST_URL = "https://www.kortennis.or.kr/cmpt/92/cmptList.do"


def collect(refresh=False):
    this_year = date.today().year
    rows = []

    for year in [this_year, this_year + 1]:
        try:
            html = kta.fetch_year_html(year, use_cache=not refresh)
            for r in kta.parse_html(html, year):
                rows.append({
                    "source": "KTA", "source_id": "", "name": r["name"],
                    "start_date": r["start_date"], "end_date": r["end_date"],
                    "venue": r["venue"], "category": r["section"], "url": KTA_LIST_URL,
                })
        except (ValueError, RuntimeError) as e:
            if year == this_year:
                raise
            print(f"  KTA {year}: 건너뜀 ({e})")

    for year in [this_year, this_year + 1]:
        html = kato.fetch_year_html(year, use_cache=not refresh)
        if html is None:
            continue
        for r in kato.parse_html(html):
            r["category"] = "동호인"
            rows.append(r)

    return rows


def refresh_if_stale(max_age_hours=6):
    """DB가 비었거나 마지막 수집이 오래됐을 때만 실제로 사이트에 요청해서 갱신"""
    last = store.latest_seen()
    if last:
        age_h = (datetime.now() - datetime.fromisoformat(last)).total_seconds() / 3600
        if age_h < max_age_hours:
            return None
    stats = store.upsert(collect(refresh=True))
    print(f"[crawler] 갱신 완료: {stats}")
    return stats


if __name__ == "__main__":
    refresh = "--refresh" in sys.argv
    rows = collect(refresh)
    print(f"수집 {len(rows)}건 (출처별: {dict(Counter(r['source'] for r in rows))})")

    stats = store.upsert(rows)
    print(f"DB 반영: 신규 {stats['new']} / 변경 {stats['changed']} / 변화없음 {stats['unchanged']}")

    upcoming = store.query_tournaments(upcoming_only=True)
    print(f"\n앞으로 열리거나 진행 중인 대회: {len(upcoming)}건 (상태: {dict(Counter(t['phase'] for t in upcoming))})")
    for t in upcoming[:5]:
        print(f"  {t['start_date']}~{t['end_date']} [{t['source']}] {t['name']} | {t['divisions'] or t['venue']} | {t['registration'] or '-'}")
    print(f"\nDB 파일: {store.DB_PATH}")