import sqlite3
from datetime import datetime, timedelta

import pytest

import accounts


@pytest.fixture
def user(db):
    return accounts.create_user("Tester@Example.com", "테스터", "password123")


# ---- 회원가입 ----
def test_create_user_normalizes_email(db):
    uid = accounts.create_user("  Tester@Example.COM ", "닉", "password123")
    assert uid > 0
    assert accounts.authenticate("tester@example.com", "password123")["nickname"] == "닉"


@pytest.mark.parametrize("email", ["", "no-at-sign", "a@b", "a b@c.com", "@c.com"])
def test_create_user_rejects_bad_email(db, email):
    with pytest.raises(ValueError, match="이메일"):
        accounts.create_user(email, "닉", "password123")


@pytest.mark.parametrize("nickname", ["", "   ", "x" * 21])
def test_create_user_rejects_bad_nickname(db, nickname):
    with pytest.raises(ValueError, match="닉네임"):
        accounts.create_user("a@b.com", nickname, "password123")


def test_create_user_rejects_short_password(db):
    with pytest.raises(ValueError, match="8자"):
        accounts.create_user("a@b.com", "닉", "short")


def test_create_user_rejects_duplicate_email_case_insensitively(user):
    with pytest.raises(ValueError, match="이미 가입"):
        accounts.create_user("TESTER@example.com", "다른닉", "password123")


# ---- 비밀번호 ----
def test_password_is_not_stored_in_plaintext_and_salted(db):
    accounts.create_user("a@b.com", "A", "password123")
    accounts.create_user("c@d.com", "C", "password123")
    with sqlite3.connect(db) as c:
        hashes = [r[0] for r in c.execute("SELECT password_hash FROM users")]
    assert all("password123" not in h for h in hashes)
    assert hashes[0] != hashes[1]


def test_authenticate_success_and_failures(user):
    assert accounts.authenticate("tester@example.com", "password123")["id"] == user
    assert accounts.authenticate("tester@example.com", "wrong-password") is None
    assert accounts.authenticate("nobody@example.com", "password123") is None


# ---- 세션 ----
def test_session_roundtrip_and_logout(user):
    token = accounts.create_session(user)
    me = accounts.user_from_token(token)
    assert me["email"] == "tester@example.com"
    accounts.delete_session(token)
    assert accounts.user_from_token(token) is None


def test_invalid_or_missing_token_returns_none(db):
    assert accounts.user_from_token(None) is None
    assert accounts.user_from_token("") is None
    assert accounts.user_from_token("not-a-real-token") is None


def test_token_is_stored_only_as_hash(user, db):
    token = accounts.create_session(user)
    with sqlite3.connect(db) as c:
        stored = [r[0] for r in c.execute("SELECT token_hash FROM login_sessions")]
    assert token not in stored


def test_expired_session_is_rejected_and_purged_on_next_login(user, db):
    token = accounts.create_session(user)
    past = (datetime.now() - timedelta(days=1)).isoformat(timespec="seconds")
    with sqlite3.connect(db) as c:
        c.execute("UPDATE login_sessions SET expires_at=?", (past,))
    assert accounts.user_from_token(token) is None
    accounts.create_session(user)
    with sqlite3.connect(db) as c:
        assert c.execute("SELECT COUNT(*) FROM login_sessions").fetchone()[0] == 1


# ---- 분석 기록 ----
STATS = {"player1": {"total_distance_m": 33.83, "avg_speed_mps": 3.41, "max_speed_mps": 14.6}}


def test_save_analysis_upserts_same_video(user):
    first = accounts.save_analysis(user, "vid-1", "2099-01-01", "처음", STATS)
    second = accounts.save_analysis(user, "vid-1", "2099-02-02", "수정", STATS)
    assert first == second
    items = accounts.list_analyses(user)
    assert len(items) == 1
    assert items[0]["title"] == "수정" and items[0]["recorded_date"] == "2099-02-02"
    assert items[0]["stats"] == STATS
    assert items[0]["heatmap_url"] == "/results/vid-1/heatmap.png"


def test_save_analysis_rejects_bad_date_and_truncates_title(user):
    with pytest.raises(ValueError):
        accounts.save_analysis(user, "v", "2099/01/01", "t", STATS)
    accounts.save_analysis(user, "v", "2099-01-01", "가" * 100, STATS)
    assert len(accounts.list_analyses(user)[0]["title"]) == 60


def test_list_analyses_sorted_by_recorded_date_desc(user):
    accounts.save_analysis(user, "a", "2099-01-01", "old", STATS)
    accounts.save_analysis(user, "b", "2099-03-01", "new", STATS)
    accounts.save_analysis(user, "c", "2099-02-01", "mid", STATS)
    assert [i["title"] for i in accounts.list_analyses(user)] == ["new", "mid", "old"]


def test_analyses_are_isolated_between_users(user):
    other = accounts.create_user("other@example.com", "남", "password123")
    aid = accounts.save_analysis(user, "v", "2099-01-01", "내 기록", STATS)
    assert accounts.list_analyses(other) == []
    assert accounts.get_analysis(other, aid) is None
    assert accounts.delete_analysis(other, aid) is False
    assert accounts.get_analysis(user, aid)["title"] == "내 기록"


def test_delete_analysis_removes_own_record_once(user):
    aid = accounts.save_analysis(user, "v", "2099-01-01", "t", STATS)
    assert accounts.delete_analysis(user, aid) is True
    assert accounts.delete_analysis(user, aid) is False
    assert accounts.list_analyses(user) == []



# ---- 로그인 연장 (버튼으로 3시간) ----
def _expires(db):
    with sqlite3.connect(db) as c:
        return c.execute("SELECT expires_at FROM login_sessions").fetchone()[0]


def test_session_lasts_one_day(user, db):
    accounts.create_session(user)
    left = datetime.fromisoformat(_expires(db)) - datetime.now()
    assert timedelta(hours=23) < left <= timedelta(days=1)


def test_extend_session_adds_three_hours_to_current_expiry(user, db):
    token = accounts.create_session(user)
    before = datetime.fromisoformat(_expires(db))
    new = accounts.extend_session(token)
    assert datetime.fromisoformat(new) - before == timedelta(hours=3)
    assert _expires(db) == new


def test_extend_session_ignores_expired_missing_and_unknown(user, db):
    token = accounts.create_session(user)
    past = (datetime.now() - timedelta(minutes=1)).isoformat(timespec="seconds")
    with sqlite3.connect(db) as c:
        c.execute("UPDATE login_sessions SET expires_at=?", (past,))
    assert accounts.extend_session(token) is None
    assert accounts.extend_session(None) is None
    assert accounts.extend_session("unknown-token") is None
    assert _expires(db) == past


def test_user_from_token_includes_expires_at(user):
    token = accounts.create_session(user)
    assert accounts.user_from_token(token)["expires_at"]
