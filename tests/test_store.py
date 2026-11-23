from datetime import date

from crawler import store


def row(**over):
    base = {
        "source": "KATO", "source_id": "100", "name": "테스트 오픈",
        "start_date": "2099-05-01", "end_date": "2099-05-03",
        "venue": "", "divisions": "남자부,여자부", "registration": "",
        "ranking_group": "A", "category": "동호인", "url": "https://example.com/100",
    }
    base.update(over)
    return base


def changes(db_path):
    con = store.connect()
    try:
        return [dict(r) for r in con.execute("SELECT * FROM tournament_changes ORDER BY id")]
    finally:
        con.close()


def test_upsert_new_row_is_counted_and_queryable(db):
    stats = store.upsert([row()])
    assert stats == {"new": 1, "changed": 0, "unchanged": 0}
    items = store.query_tournaments()
    assert len(items) == 1
    assert items[0]["name"] == "테스트 오픈"


def test_upsert_same_row_twice_is_unchanged_without_history(db):
    store.upsert([row()])
    stats = store.upsert([row()])
    assert stats == {"new": 0, "changed": 0, "unchanged": 1}
    assert changes(db) == []


def test_upsert_records_changed_field_with_old_and_new_value(db):
    store.upsert([row(registration="")])
    stats = store.upsert([row(registration="접수중")])
    assert stats["changed"] == 1
    hist = changes(db)
    assert len(hist) == 1
    assert hist[0]["field"] == "registration"
    assert hist[0]["old_value"] == ""
    assert hist[0]["new_value"] == "접수중"
    assert store.query_tournaments()[0]["registration"] == "접수중"


def test_upsert_multiple_changed_fields_create_one_history_row_each(db):
    store.upsert([row()])
    store.upsert([row(name="이름 변경", venue="서울")])
    assert {c["field"] for c in changes(db)} == {"name", "venue"}


def test_missing_value_and_empty_string_are_not_a_change(db):
    store.upsert([row(venue="")])
    stats = store.upsert([row(venue=None)])
    assert stats["unchanged"] == 1
    assert changes(db) == []


def test_untracked_field_update_is_not_recorded_as_change(db):
    store.upsert([row(url="https://example.com/a")])
    stats = store.upsert([row(url="https://example.com/b")])
    assert stats["unchanged"] == 1
    assert changes(db) == []
    assert store.query_tournaments()[0]["url"] == "https://example.com/b"


def test_make_uid_uses_source_id_when_present():
    assert store.make_uid({"source": "KATO", "source_id": "42", "name": "x", "start_date": "d"}) == "KATO:42"


def test_make_uid_falls_back_to_stable_hash_without_source_id():
    r = {"source": "KTA", "source_id": "", "name": "대회", "start_date": "2099-01-01"}
    uid = store.make_uid(r)
    assert uid.startswith("KTA:") and len(uid) == len("KTA:") + 12
    assert uid == store.make_uid(dict(r))
    assert uid != store.make_uid({**r, "name": "다른 대회"})


def test_compute_phase_boundaries():
    today = date(2099, 6, 15)
    assert store.compute_phase("2099-06-01", "2099-06-14", today) == "종료"
    assert store.compute_phase("2099-06-16", "2099-06-20", today) == "예정"
    assert store.compute_phase("2099-06-15", "2099-06-15", today) == "진행중"
    assert store.compute_phase("2099-06-10", "2099-06-15", today) == "진행중"
    assert store.compute_phase("2099-06-15", "2099-06-20", today) == "진행중"


def test_query_excludes_past_by_default_and_includes_when_requested(db):
    store.upsert([
        row(source_id="1", name="지난 대회", start_date="2000-01-01", end_date="2000-01-02"),
        row(source_id="2", name="다음 대회", start_date="2099-01-01", end_date="2099-01-02"),
    ])
    assert [t["name"] for t in store.query_tournaments()] == ["다음 대회"]
    assert {t["name"] for t in store.query_tournaments(upcoming_only=False)} == {"지난 대회", "다음 대회"}
    past = [t for t in store.query_tournaments(upcoming_only=False) if t["name"] == "지난 대회"][0]
    assert past["phase"] == "종료"


def test_query_filters_by_source_and_division_and_sorts_by_start(db):
    store.upsert([
        row(source_id="1", name="B", start_date="2099-03-01", end_date="2099-03-02", divisions="여자부"),
        row(source_id="2", name="A", start_date="2099-02-01", end_date="2099-02-02", divisions="남자부"),
        row(source="KTA", source_id="", name="C", start_date="2099-04-01", end_date="2099-04-02", divisions=""),
    ])
    assert [t["name"] for t in store.query_tournaments()] == ["A", "B", "C"]
    assert [t["name"] for t in store.query_tournaments(source="KTA")] == ["C"]
    assert [t["name"] for t in store.query_tournaments(division="여자")] == ["B"]
    assert len(store.query_tournaments(limit=2)) == 2


def test_latest_seen_is_none_for_empty_db_then_set_after_upsert(db):
    assert store.latest_seen() is None
    store.upsert([row()])
    assert store.latest_seen() is not None
