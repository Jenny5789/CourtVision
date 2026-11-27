import numpy as np
import pytest

from ball_tracker import DETECTED, INTERPOLATED, NONE, TrackerConfig, track_ball

W = 1920
T = 40


def truth(n=T):
    t = np.arange(n)
    return np.stack([200 + 25.0 * t, 700 - 30.0 * t + 1.0 * t * t], axis=1)


def frames_from(points, conf=0.9, drop=(), extra=None):
    dets = []
    for t, (x, y) in enumerate(points):
        rows = [] if t in drop else [[x, y, conf]]
        if extra:
            rows += extra(t)
        dets.append(np.array(rows, dtype=np.float32).reshape(-1, 3))
    return dets


def test_clean_track_is_all_detected_and_exact():
    pts = truth()
    out = track_ball(frames_from(pts), W)
    assert (out[:, 3] == DETECTED).all()
    assert np.allclose(out[:, :2], pts, atol=1e-3)


def test_missing_frames_are_interpolated_close_to_truth():
    pts = truth()
    drop = {15, 16, 17}
    out = track_ball(frames_from(pts, drop=drop), W)
    for t in drop:
        assert out[t, 3] == INTERPOLATED
        assert np.hypot(*(out[t, :2] - pts[t])) < 6
    assert (np.delete(out[:, 3], list(drop)) == DETECTED).all()


def test_gap_longer_than_max_gap_is_not_bridged():
    pts = truth()
    drop = set(range(15, 15 + 12))
    out = track_ball(frames_from(pts, drop=drop), W, TrackerConfig(max_gap=8))
    assert (out[list(drop), 3] == NONE).all()


def test_isolated_false_detections_are_ignored():
    rng = np.random.default_rng(0)
    dets = [np.zeros((0, 3), np.float32) for _ in range(T)]
    for t in (5, 14, 23, 33):
        dets[t] = np.array([[rng.uniform(100, 1800), rng.uniform(100, 1000), 0.6]], np.float32)
    out = track_ball(dets, W)
    assert (out[:, 3] == NONE).all()


def test_short_chain_below_min_nodes_is_dropped():
    pts = truth(2)
    out = track_ball(frames_from(pts, conf=0.95), W, TrackerConfig(min_nodes=3))
    assert (out[:, 3] == NONE).all()


def test_follows_moving_ball_not_stronger_stationary_distractor():
    pts = truth()
    distractor = lambda t: [[1500.0, 900.0, 0.8]]
    out = track_ball(frames_from(pts, conf=0.7, extra=distractor), W)
    near_ball = np.hypot(*(out[:, :2] - pts).T) < 5
    assert near_ball.mean() > 0.9
    assert not (np.hypot(out[:, 0] - 1500, out[:, 1] - 900) < 5).any()


def test_impossible_jump_breaks_the_track_instead_of_interpolating():
    left = truth(15)
    right = np.stack([1700.0 + 5 * np.arange(15), 300.0 + np.zeros(15)], axis=1)
    pts = np.vstack([left, np.full((2, 2), np.nan), right])
    dets = frames_from(np.nan_to_num(pts), drop={15, 16})
    out = track_ball(dets, W, TrackerConfig(max_speed=0.12))
    assert (out[15:17, 3] == NONE).all()
    assert (out[:15, 3] == DETECTED).all() and (out[17:, 3] == DETECTED).all()


def test_speed_gate_blocks_links_faster_than_max_speed():
    # 속도가 상한을 살짝(4%) 넘는 직선 궤적. 점수가 높아 비용만으로는 연결이 이득이지만, 상한을 넘는 연결은 아예 허용하지 않는다.
    t = np.arange(T)
    pts = np.stack([200 + 20.0 * t, 500.0 + 0 * t], axis=1)
    out = track_ball(frames_from(pts, conf=0.99), W, TrackerConfig(max_speed=0.01))
    assert (out[:, 3] == NONE).all()
    out = track_ball(frames_from(pts, conf=0.99), W, TrackerConfig(max_speed=0.12))
    assert (out[:, 3] == DETECTED).all()


def test_chain_without_a_confident_anchor_is_dropped():
    pts = truth()
    weak = frames_from(pts, conf=0.4)
    assert (track_ball(weak, W, TrackerConfig(min_peak_conf=0.5))[:, 3] == NONE).all()
    assert (track_ball(weak, W, TrackerConfig(min_peak_conf=0.0))[:, 3] == DETECTED).all()
    anchored = frames_from(pts, conf=0.4)
    anchored[20] = np.array([[*pts[20], 0.9]], np.float32)
    assert (track_ball(anchored, W, TrackerConfig(min_peak_conf=0.5))[:, 3] == DETECTED).all()


def test_low_confidence_tails_are_trimmed_but_interior_is_kept():
    pts = truth()
    dets = frames_from(pts, conf=0.9)
    for t in list(range(0, 6)) + list(range(34, 40)):      # 앞·뒤 끝은 점수가 낮은 검출
        dets[t] = np.array([[*pts[t], 0.25]], np.float32)
    dets[20] = np.array([[*pts[20], 0.25]], np.float32)     # 중간의 낮은 점수 검출은 이웃이 받쳐 주므로 유지
    out = track_ball(dets, W, TrackerConfig(trim_conf=0.5))
    assert (out[:6, 3] == NONE).all() and (out[34:, 3] == NONE).all()
    assert (out[6:34, 3] == DETECTED).all()
    untrimmed = track_ball(dets, W, TrackerConfig(trim_conf=0.0))
    assert (untrimmed[:, 3] == DETECTED).all()


def test_empty_inputs():
    assert track_ball([], W).shape == (0, 4)
    out = track_ball([np.zeros((0, 3), np.float32)] * 10, W)
    assert out.shape == (10, 4) and (out[:, 3] == NONE).all()
