import pytest
from fastapi.testclient import TestClient

import accounts
import main
from crawler import store

EMAIL, PASSWORD = "api@example.com", "password123"


@pytest.fixture
def client(db):
    main.sessions.clear()
    return TestClient(main.app)          # with 문 없이 만들어 lifespan(크롤러 스레드)을 실행하지 않음


def signup(client, email=EMAIL, nickname="닉"):
    return client.post("/api/signup", json={"email": email, "nickname": nickname, "password": PASSWORD})


# ---- 정적 화면 ----
def test_root_serves_frontend(client):
    r = client.get("/")
    assert r.status_code == 200 and "CourtVision" in r.text


# ---- 인증 ----
def test_signup_sets_cookie_and_logs_in(client):
    r = signup(client)
    assert r.status_code == 200 and r.json()["email"] == EMAIL
    assert "httponly" in r.headers["set-cookie"].lower()
    assert client.get("/api/me").json()["user"]["email"] == EMAIL


def test_signup_validation_and_duplicate_return_400(client):
    assert signup(client, email="bad").status_code == 400
    assert client.post("/api/signup", json={"email": "a@b.com", "nickname": "n", "password": "short"}).status_code == 400
    assert signup(client).status_code == 200
    assert signup(TestClient(main.app)).status_code == 400


def test_login_wrong_password_is_401_and_correct_is_200(client):
    signup(client)
    other = TestClient(main.app)
    assert other.post("/api/login", json={"email": EMAIL, "password": "wrong-password"}).status_code == 401
    assert other.post("/api/login", json={"email": EMAIL, "password": PASSWORD}).status_code == 200
    assert other.get("/api/me").json()["user"]["email"] == EMAIL


def test_logout_invalidates_session(client):
    signup(client)
    assert client.post("/api/logout").json() == {"ok": True}
    assert client.get("/api/me").json() == {"user": None}


def test_protected_endpoints_require_login(client):
    assert client.get("/api/analyses").status_code == 401
    assert client.delete("/api/analyses/1").status_code == 401


# ---- 분석 기록 ----
STATS = {"player1": {"total_distance_m": 1.0, "avg_speed_mps": 1.0, "max_speed_mps": 1.0}}


def test_list_and_delete_own_analysis_and_not_others(client):
    signup(client)
    me = accounts.authenticate(EMAIL, PASSWORD)["id"]
    aid = accounts.save_analysis(me, "v1", "2099-01-01", "기록", STATS)

    assert [i["id"] for i in client.get("/api/analyses").json()["items"]] == [aid]

    intruder = TestClient(main.app)
    signup(intruder, email="intruder@example.com")
    assert intruder.delete(f"/api/analyses/{aid}").status_code == 404

    assert client.delete(f"/api/analyses/{aid}").status_code == 200
    assert client.delete(f"/api/analyses/{aid}").status_code == 404


# ---- 대회 ----
def test_tournaments_endpoint_filters_and_reports_count(client):
    store.upsert([
        {"source": "KATO", "source_id": "1", "name": "A", "start_date": "2099-01-01", "end_date": "2099-01-02"},
        {"source": "KTA", "source_id": "", "name": "B", "start_date": "2099-02-01", "end_date": "2099-02-02"},
        {"source": "KATO", "source_id": "3", "name": "지난", "start_date": "2000-01-01", "end_date": "2000-01-02"},
    ])
    body = client.get("/api/tournaments").json()
    assert body["count"] == 2 and body["last_crawled"]
    assert [t["name"] for t in client.get("/api/tournaments", params={"source": "KTA"}).json()["items"]] == ["B"]
    assert client.get("/api/tournaments", params={"upcoming": "false"}).json()["count"] == 3


# ---- 영상 업로드·보정 흐름(검출 모델 없이 가능한 구간) ----
def test_upload_then_set_frame(client, make_video):
    path = make_video(frames=30)
    with open(path, "rb") as f:
        r = client.post("/api/upload", files={"file": ("clip.mp4", f, "video/mp4")})
    assert r.status_code == 200
    vid = r.json()["video_id"]
    assert client.get(r.json()["frame_url"]).status_code == 200
    assert client.post("/api/set_frame", json={"video_id": vid, "time_sec": 0.5}).status_code == 200


def test_unknown_video_id_is_404_for_set_frame_calibrate_analyze(client):
    assert client.post("/api/set_frame", json={"video_id": "x", "time_sec": 0}).status_code == 404
    assert client.post("/api/calibrate", json={"video_id": "x", "corners": [[0, 0]] * 4}).status_code == 404
    assert client.post("/api/analyze", json={"video_id": "x"}).status_code == 404


def test_calibrate_requires_exactly_four_corners(client):
    main.sessions["v"] = {"video_path": "unused", "output_dir": "unused"}
    r = client.post("/api/calibrate", json={"video_id": "v", "corners": [[0, 0], [1, 1], [2, 2]]})
    assert r.status_code == 400



# ---- 로그인 연장 (버튼으로 3시간) ----
def test_me_reports_expiry_and_extend_adds_three_hours(client):
    from datetime import datetime, timedelta
    signup(client)
    before = datetime.fromisoformat(client.get("/api/me").json()["user"]["expires_at"])
    r = client.post("/api/session/extend")
    assert r.status_code == 200
    assert datetime.fromisoformat(r.json()["expires_at"]) - before == timedelta(hours=3)
    assert "cv_session" in r.headers["set-cookie"] and "httponly" in r.headers["set-cookie"].lower()
    assert client.get("/api/me").json()["user"]["expires_at"] == r.json()["expires_at"]


def test_extend_requires_valid_login_and_api_use_does_not_extend(client):
    assert client.post("/api/session/extend").status_code == 401
    signup(client)
    before = client.get("/api/me").json()["user"]["expires_at"]
    client.get("/api/analyses")
    assert client.get("/api/me").json()["user"]["expires_at"] == before   # 자동 연장 없음
    client.post("/api/logout")
    assert client.post("/api/session/extend").status_code == 401
