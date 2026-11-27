"""공 추적 서비스 연결(pipeline.run_ball_analysis, /api/analyze 의 ball 항목) 테스트.

공 검출 모델은 쓰지 않는다. 모델 호출 부분(_get_ball_model, _ball_detect_frames)을 가짜로 바꿔
"후보 -> 추적 -> 그림·CSV·응답" 흐름과 "공 추적이 실패해도 선수 분석은 돌려준다"는 약속을 검증한다.
"""
import os

import cv2
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import main
import pipeline

CORNERS = [[690, 610], [1230, 610], [1130, 435], [795, 435]]
FRAMES = 30


def ball_dets(missing=(), outlier=True):
    """직선으로 움직이는 공(확신도 0.8)과, 멀리 떨어진 낮은 확신도 오검출 후보를 프레임마다 만든다."""
    dets = []
    for t in range(FRAMES):
        rows = [] if t in missing else [[200 + 20 * t, 500, 0.8]]
        if outlier and t == 5:
            rows.append([1500, 100, 0.2])
        dets.append(np.array(rows, np.float32).reshape(-1, 3))
    return dets


@pytest.fixture
def fake_ball(monkeypatch):
    """가짜 공 검출 결과를 심는 함수."""
    def _set(dets):
        monkeypatch.setattr(pipeline, "_get_ball_model", lambda: object())
        monkeypatch.setattr(pipeline, "_ball_detect_frames", lambda video_path, model: (dets, 1920))
    return _set


def test_ball_analysis_is_off_without_weights(monkeypatch, make_video, tmp_path):
    monkeypatch.setattr(pipeline, "BALL_MODEL_PATH", str(tmp_path / "missing.pt"))
    monkeypatch.setattr(pipeline, "_ball_model", None)
    assert pipeline.run_ball_analysis(make_video(frames=3), str(tmp_path / "out")) is None


def test_ball_analysis_can_be_switched_off(monkeypatch, make_video, tmp_path):
    monkeypatch.setattr(pipeline, "BALL_ENABLED", False)
    assert pipeline.run_ball_analysis(make_video(frames=3), str(tmp_path / "out")) is None


def test_ball_analysis_tracks_ball_drops_outlier_and_saves_outputs(fake_ball, make_video, tmp_path):
    fake_ball(ball_dets())
    out_dir = tmp_path / "out"
    result = pipeline.run_ball_analysis(make_video(frames=3), str(out_dir))

    assert result["total_frames"] == FRAMES and result["tracked_frames"] == FRAMES
    assert cv2.imread(result["image"]) is not None
    csv = pd.read_csv(out_dir / "ball_tracking.csv")
    assert list(csv.columns) == ["frame", "x", "y", "conf", "state"]
    assert (csv["state"] == 1).all()                         # 전부 실제 검출, 보간 없음
    assert csv["x"].between(200, 800).all()                  # 멀리 떨어진 오검출(x=1500)은 채택되지 않음


def test_ball_analysis_fills_short_gap_by_interpolation(fake_ball, make_video, tmp_path):
    fake_ball(ball_dets(missing=(10, 11), outlier=False))
    out_dir = tmp_path / "out"
    result = pipeline.run_ball_analysis(make_video(frames=3), str(out_dir))

    csv = pd.read_csv(out_dir / "ball_tracking.csv")
    assert result["tracked_frames"] == FRAMES
    assert csv.loc[[10, 11], "state"].tolist() == [2, 2]     # 검출 없는 2프레임은 보간으로 채움
    assert csv.loc[10, "x"] == pytest.approx(200 + 20 * 10, abs=1)


def test_trajectory_image_does_not_join_points_that_are_far_apart(fake_ball, make_video, tmp_path):
    """프레임 14->15 에서 공이 160px 건너뛰면(추적은 같은 공으로 보지만) 그림에서는 선으로 잇지 않는다."""
    dets = [np.array([[200 + 10 * t + (150 if t >= 15 else 0), 500, 0.8]], np.float32) for t in range(FRAMES)]
    fake_ball(dets)
    result = pipeline.run_ball_analysis(make_video(frames=3), str(tmp_path / "out"))
    assert result["tracked_frames"] == FRAMES                  # 추적은 하나의 공으로 이어짐

    img = cv2.imread(result["image"])
    assert img[500, 300].max() > 100                           # 가까운 점 사이(t=9~10)는 선으로 이어짐
    assert img[500, 420].max() < 30                            # 160px 떨어진 두 점 사이 중간은 비어 있음


def test_ball_analysis_with_no_candidates_reports_zero_tracked(fake_ball, make_video, tmp_path):
    fake_ball([np.zeros((0, 3), np.float32)] * FRAMES)
    result = pipeline.run_ball_analysis(make_video(frames=3), str(tmp_path / "out"))
    assert result["tracked_frames"] == 0 and result["total_frames"] == FRAMES
    assert cv2.imread(result["image"]) is not None            # 공이 없어도 그림 파일은 만들어짐


# ---- /api/analyze 응답 ----
@pytest.fixture
def analyze_ready(db, make_video, tmp_path):
    """코트 보정까지 끝난 상태의 세션과 클라이언트."""
    main.sessions.clear()
    rows = [{"frame": f, "time": f / 30, "track_id": 1, "x1": 800 + f, "y1": 300, "x2": 900 + f, "y2": 600}
            for f in range(FRAMES)]
    main.sessions["v"] = {"video_path": make_video(frames=FRAMES), "output_dir": str(tmp_path / "res"),
                          "H": pipeline.compute_homography(CORNERS), "full_tracks": pd.DataFrame(rows),
                          "fps": 30.0, "candidates": [], "player1_id": 1}
    return TestClient(main.app)


def test_analyze_includes_ball_result_when_available(analyze_ready, monkeypatch):
    monkeypatch.setattr(pipeline, "run_ball_analysis",
                        lambda video_path, output_dir: {"image": "x.jpg", "tracked_frames": 25, "total_frames": 30})
    body = analyze_ready.post("/api/analyze", json={"video_id": "v"}).json()
    assert body["ball"] == {"image_url": "/results/v/ball_trajectory.jpg", "tracked_frames": 25, "total_frames": 30}
    assert "player1" in body["stats"]


def test_analyze_returns_player_stats_even_if_ball_tracking_fails(analyze_ready, monkeypatch):
    def boom(video_path, output_dir):
        raise RuntimeError("공 모델 오류")
    monkeypatch.setattr(pipeline, "run_ball_analysis", boom)
    r = analyze_ready.post("/api/analyze", json={"video_id": "v"})
    assert r.status_code == 200
    assert r.json()["ball"] is None and "player1" in r.json()["stats"]


def test_analyze_covers_only_player1_even_if_a_second_player_is_clicked(analyze_ready, monkeypatch):
    """분석 대상은 카메라에서 가까운 선수(player1) 한 명뿐. 예전 화면이 두 번째 선수 클릭 좌표를 보내도 무시한다."""
    monkeypatch.setattr(pipeline, "run_ball_analysis", lambda video_path, output_dir: None)
    far = pd.DataFrame([{"frame": f, "time": f / 30, "track_id": 2, "x1": 900, "y1": 250, "x2": 960, "y2": 340}
                        for f in range(FRAMES)])
    session = main.sessions["v"]
    session["full_tracks"] = pd.concat([session["full_tracks"], far], ignore_index=True)
    session["candidates"] = [{"track_id": 2, "x1": 900, "y1": 250, "x2": 960, "y2": 340}]
    body = analyze_ready.post("/api/analyze", json={"video_id": "v", "click_x": 930, "click_y": 300}).json()
    assert list(body["stats"]) == ["player1"]


def test_my_page_list_gives_ball_url_only_when_the_ball_image_exists(db):
    """마이페이지 목록: 공 궤적 그림이 있는 기록에만 ball_url, 없으면 None."""
    import accounts
    client = TestClient(main.app)
    client.post("/api/signup", json={"email": "ball@example.com", "nickname": "닉", "password": "password123"})
    uid = accounts.authenticate("ball@example.com", "password123")["id"]
    stats = {"player1": {"total_distance_m": 1.0, "avg_speed_mps": 1.0, "max_speed_mps": 1.0}}
    accounts.save_analysis(uid, "with-ball", "2099-01-02", "공 있음", stats)
    accounts.save_analysis(uid, "no-ball", "2099-01-01", "공 없음", stats)
    folder = os.path.join(main.RESULTS_DIR, "with-ball")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "ball_trajectory.jpg"), "wb") as f:
        f.write(b"x")

    by_title = {i["title"]: i for i in client.get("/api/analyses").json()["items"]}
    assert by_title["공 있음"]["ball_url"] == "/results/with-ball/ball_trajectory.jpg"
    assert by_title["공 없음"]["ball_url"] is None
    assert "heatmap_url" in by_title["공 없음"]                         # 기존 항목은 그대로


def test_analyze_ball_is_none_when_feature_is_off(analyze_ready, monkeypatch):
    monkeypatch.setattr(pipeline, "run_ball_analysis", lambda video_path, output_dir: None)
    assert analyze_ready.post("/api/analyze", json={"video_id": "v"}).json()["ball"] is None
