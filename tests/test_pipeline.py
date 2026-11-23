"""좌표 변환·분석 통계 검증.

실제 영상이 아니라 "가상 카메라"로 만든 합성 데이터로 검증한다:
코트 규격(m) -> 가상 카메라 픽셀 -> compute_homography -> 다시 미터로 되돌렸을 때 오차를 본다.
실제 촬영 영상에서의 정확도 검증은 아니다(그건 별도 과제).
"""
import itertools

import cv2
import numpy as np
import pandas as pd
import pytest

import pipeline

# 테니스 코트 공식 규격(m). pipeline 의 상수를 그대로 쓰면 상수가 틀려도 테스트가 같이 틀려 통과하므로 독립적으로 적는다.
COURT_W, COURT_L, SERVICE, NET = 8.23, 23.77, 5.485, 11.885
FULL_COURT = np.array([[0, 0], [COURT_W, 0], [COURT_W, COURT_L], [0, COURT_L]], dtype=np.float32)
CLICK_POINTS_M = np.array([[0, SERVICE], [COURT_W, SERVICE], [COURT_W, NET], [0, NET]], dtype=np.float32)

# 코트 4모서리(m) -> 가상 카메라 픽셀(사다리꼴: 가까운 베이스라인이 화면 아래, 넓게 보임)
CAMERA_PIXELS = np.array([[300, 1000], [1620, 1000], [1220, 250], [700, 250]], dtype=np.float32)
H_TRUE = cv2.getPerspectiveTransform(FULL_COURT, CAMERA_PIXELS)  # 미터 -> 픽셀


def to_pixels(points_m):
    arr = np.array(points_m, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(arr, H_TRUE).reshape(-1, 2)


def to_meters(H, points_px):
    arr = np.array(points_px, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(arr, H).reshape(-1, 2)


@pytest.fixture
def clicked_corners():
    return to_pixels(CLICK_POINTS_M).tolist()


def test_pipeline_court_constants_match_official_dimensions():
    assert np.allclose(pipeline.REAL_COURT_POINTS, CLICK_POINTS_M)
    assert np.allclose(pipeline.FULL_COURT_POINTS, FULL_COURT)
    assert NET - SERVICE == pytest.approx(6.40)   # 서비스라인 ~ 네트 공식 거리


# ---- 클릭 순서 ----
def test_order_court_points_is_independent_of_click_order(clicked_corners):
    expected = pipeline.order_court_points(clicked_corners)
    for perm in itertools.permutations(clicked_corners):
        assert pipeline.order_court_points(list(perm)) == expected
    near_left, near_right, far_right, far_left = expected
    assert near_left[1] > far_left[1] and near_left[0] < near_right[0] and far_left[0] < far_right[0]


# ---- Homography ----
def test_homography_recovers_known_court_distances(clicked_corners):
    H = pipeline.compute_homography(clicked_corners)
    court_m = to_meters(H, to_pixels(FULL_COURT))
    assert np.allclose(court_m, FULL_COURT, atol=0.01)

    # 서비스라인(5.485) ~ 네트(11.885) 거리 = 6.4m, 싱글 코트 폭 = 8.23m, 베이스라인 간 = 23.77m
    p = to_meters(H, to_pixels([[0, 5.485], [0, 11.885], [0, 0], [8.23, 0], [0, 23.77]]))
    assert np.linalg.norm(p[1] - p[0]) == pytest.approx(6.4, abs=0.02)
    assert np.linalg.norm(p[3] - p[2]) == pytest.approx(8.23, abs=0.02)
    assert np.linalg.norm(p[4] - p[2]) == pytest.approx(23.77, abs=0.05)


def test_homography_is_independent_of_click_order(clicked_corners):
    base = pipeline.compute_homography(clicked_corners)
    shuffled = pipeline.compute_homography(list(reversed(clicked_corners)))
    assert np.allclose(base, shuffled, atol=1e-6)


def test_homography_tolerates_small_click_error(clicked_corners):
    """클릭이 3px 어긋나도 코트 위 거리 오차가 크지 않은지(참고용 상한)"""
    rng = np.random.default_rng(0)
    noisy = (np.array(clicked_corners) + rng.uniform(-3, 3, (4, 2))).tolist()
    H = pipeline.compute_homography(noisy)
    p = to_meters(H, to_pixels([[0, 5.485], [0, 11.885]]))
    assert np.linalg.norm(p[1] - p[0]) == pytest.approx(6.4, abs=0.5)


def test_court_projection_roundtrip_and_overlay(clicked_corners):
    H = pipeline.compute_homography(clicked_corners)
    back = pipeline._project_to_pixels(H, CLICK_POINTS_M)
    assert np.allclose(back, pipeline.order_court_points(clicked_corners), atol=0.5)
    segs = pipeline.court_overlay_segments(H)
    assert len(segs) == len(pipeline.COURT_LINES_M) and all(len(s) == 4 for s in segs)
    assert len(pipeline.court_polygon_pixels(H)) == 4


# ---- 후보 선수 ----
def tracks(rows):
    return pd.DataFrame(rows, columns=["frame", "time", "track_id", "x1", "y1", "x2", "y2"])


def test_candidates_pick_largest_box_inside_court_and_drop_far_outside(clicked_corners):
    H = pipeline.compute_homography(clicked_corners)
    polygon = pipeline.court_polygon_pixels(H)
    df = tracks([
        [0, 0.0, 1, 900, 700, 1000, 900],     # 코트 안, 작은 박스
        [0, 0.0, 2, 800, 500, 1000, 900],     # 코트 안, 큰 박스 -> player1
        [0, 0.0, 3, 10, 10, 50, 100],         # 코트에서 한참 벗어남 -> 제외
    ])
    candidates, player1 = pipeline.get_candidates_at_time(df, 30.0, polygon, 0.0)
    assert {c["track_id"] for c in candidates} == {1, 2}
    assert player1 == 2


def test_candidates_use_nearest_frame_and_handle_nobody(clicked_corners):
    polygon = pipeline.court_polygon_pixels(pipeline.compute_homography(clicked_corners))
    df = tracks([[10, 0.33, 7, 900, 700, 1000, 900]])
    _, player1 = pipeline.get_candidates_at_time(df, 30.0, polygon, 0.0)
    assert player1 == 7
    far = tracks([[0, 0.0, 9, 10, 10, 50, 100]])
    assert pipeline.get_candidates_at_time(far, 30.0, polygon, 0.0) == ([], None)


def test_pick_player2_by_click_position():
    cands = [{"track_id": 4, "x1": 100, "y1": 100, "x2": 200, "y2": 300}]
    assert pipeline.pick_player2(cands, 150, 200) == 4
    assert pipeline.pick_player2(cands, 500, 500) is None


# ---- 통계: 알려진 속도로 움직이는 선수 ----
def moving_player_tracks(speed_mps, seconds=3, fps=30, x_m=4.0, y_start=2.0, track_id=1):
    rows = []
    for f in range(int(seconds * fps) + 1):
        t = f / fps
        px, py = to_pixels([[x_m, y_start + speed_mps * t]])[0]
        px, py = int(round(px)), int(round(py))
        rows.append([f, round(t, 3), track_id, px - 20, py - 120, px + 20, py])
    return tracks(rows)


def test_run_full_analysis_reports_distance_and_speed_close_to_truth(clicked_corners, make_video, tmp_path):
    H = pipeline.compute_homography(clicked_corners)
    video = make_video(frames=3)
    speed, seconds = 4.0, 3
    result = pipeline.run_full_analysis(
        moving_player_tracks(speed, seconds), 30.0, {1: "player1"}, H, video, str(tmp_path / "out"))

    stats = result["stats"]["player1"]
    assert stats["total_distance_m"] == pytest.approx(speed * seconds, rel=0.08)
    assert stats["avg_speed_mps"] == pytest.approx(speed, rel=0.08)
    for key in ("csv_path", "trajectory_image", "heatmap_image"):
        assert (tmp_path / "out").exists() and result[key]
    assert pd.read_csv(result["csv_path"])["player"].eq("player1").all()


def test_run_full_analysis_stationary_player_has_near_zero_distance(clicked_corners, make_video, tmp_path):
    H = pipeline.compute_homography(clicked_corners)
    result = pipeline.run_full_analysis(
        moving_player_tracks(0.0), 30.0, {1: "player1"}, H, make_video(frames=2), str(tmp_path / "out"))
    assert result["stats"]["player1"]["total_distance_m"] < 0.5


def test_run_full_analysis_raises_for_unknown_track(clicked_corners, make_video, tmp_path):
    H = pipeline.compute_homography(clicked_corners)
    with pytest.raises(ValueError, match="추적하지 못했습니다"):
        pipeline.run_full_analysis(
            moving_player_tracks(1.0), 30.0, {99: "player1"}, H, make_video(frames=2), str(tmp_path / "out"))
