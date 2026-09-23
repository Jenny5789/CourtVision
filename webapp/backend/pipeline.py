import cv2
import numpy as np
import pandas as pd
import os
from ultralytics import YOLO

MODEL = YOLO("yolo11n.pt")
IMGSZ = int(os.environ.get("YOLO_IMGSZ", "960"))   # CPU 컨테이너에서는 값을 낮추면 빨라짐(정확도는 약간 하락)

# 사용자가 클릭하는 4점의 실제 좌표(m): 가까운 서비스라인 왼쪽 끝 -> 오른쪽 끝 -> 네트 아래 오른쪽 -> 네트 아래 왼쪽
# 좌표계: 가까운 쪽 베이스라인 y=0, 네트 y=11.885, 먼 쪽 베이스라인 y=23.77, 싱글 코트 폭 x=0~8.23
REAL_COURT_POINTS = np.array([
    [0, 5.485],
    [8.23, 5.485],
    [8.23, 11.885],
    [0, 11.885],
], dtype=np.float32)

COURT_W = 8.23
COURT_L = 23.77
FULL_COURT_POINTS = np.array([[0, 0], [COURT_W, 0], [COURT_W, COURT_L], [0, COURT_L]], dtype=np.float32)

# 보정 검증용 오버레이: 실제 코트 선(m) 선분들
COURT_LINES_M = [
    ((0, 0), (COURT_W, 0)), ((COURT_W, 0), (COURT_W, COURT_L)),
    ((COURT_W, COURT_L), (0, COURT_L)), ((0, COURT_L), (0, 0)),
    ((0, 5.485), (COURT_W, 5.485)), ((0, 18.285), (COURT_W, 18.285)),
    ((COURT_W / 2, 5.485), (COURT_W / 2, 18.285)),
    ((0, 11.885), (COURT_W, 11.885)),
]


def _project_to_pixels(H, pts_m):
    Hinv = np.linalg.inv(H)
    arr = np.array(pts_m, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(arr, Hinv).reshape(-1, 2)


def court_polygon_pixels(H):
    """전체 코트(싱글) 외곽을 픽셀 좌표로 투영 (선수 후보 필터용)"""
    return _project_to_pixels(H, FULL_COURT_POINTS).astype(int).tolist()


def court_overlay_segments(H):
    """코트 선들을 픽셀 좌표 선분 [[x1,y1,x2,y2], ...]로 투영 (화면에 겹쳐 그려 보정 확인용)"""
    segs = []
    for (a, b) in COURT_LINES_M:
        p = _project_to_pixels(H, [a, b])
        segs.append([float(p[0][0]), float(p[0][1]), float(p[1][0]), float(p[1][1])])
    return segs


def save_first_frame(video_path, output_dir, time_sec=0.0):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError("영상을 읽을 수 없습니다.")

    os.makedirs(output_dir, exist_ok=True)
    frame_path = os.path.join(output_dir, "first_frame.jpg")
    cv2.imwrite(frame_path, frame)
    return frame_path


def order_court_points(points):
    """클릭 순서와 무관하게 화면 위치로 정렬: 가까운-왼쪽, 가까운-오른쪽, 먼-오른쪽, 먼-왼쪽
    (카메라에 가까운 점일수록 화면 아래쪽 = y가 큼)"""
    by_y = sorted(points, key=lambda p: p[1])
    far = sorted(by_y[:2], key=lambda p: p[0])
    near = sorted(by_y[2:], key=lambda p: p[0])
    return [near[0], near[1], far[1], far[0]]


def compute_homography(court_corners):
    ordered = order_court_points([list(p) for p in court_corners])
    pixel_points = np.array(ordered, dtype=np.float32)
    H, _ = cv2.findHomography(pixel_points, REAL_COURT_POINTS)
    return H


def track_full_video(video_path):
    """영상 전체를 한 번만 추적해서 모든 사람의 프레임별 위치를 반환 (한 번만 실행, 재사용됨)"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    rows = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = MODEL.track(frame, classes=[0], conf=0.3, imgsz=IMGSZ,
                               tracker="botsort.yaml", persist=True, verbose=False)

        for box in results[0].boxes:
            tid = int(box.id[0]) if box.id is not None else -1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            rows.append({
                "frame": frame_idx,
                "time": round(frame_idx / fps, 3),
                "track_id": tid,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })

        frame_idx += 1

    cap.release()
    return pd.DataFrame(rows), fps


def get_candidates_at_time(full_tracks, fps, court_corners, time_sec):
    court_polygon = np.array(court_corners, dtype=np.int32)
    target_frame = round(time_sec * fps)

    # 가장 가까운 프레임 찾기
    available_frames = full_tracks["frame"].unique()
    nearest_frame = min(available_frames, key=lambda f: abs(f - target_frame))
    frame_rows = full_tracks[full_tracks["frame"] == nearest_frame]

    candidates = []
    for _, row in frame_rows.iterrows():
        x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
        bottom_center = (int((x1 + x2) / 2), int(y2))
        if cv2.pointPolygonTest(court_polygon, bottom_center, True) < -150:
            continue
        candidates.append({"track_id": int(row["track_id"]), "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)})

    player1_id = None
    if candidates:
        areas = [(c["track_id"], (c["x2"] - c["x1"]) * (c["y2"] - c["y1"])) for c in candidates]
        player1_id = max(areas, key=lambda a: a[1])[0]

    return candidates, player1_id


def pick_player2(candidates, click_x, click_y):
    for c in candidates:
        if c["x1"] <= click_x <= c["x2"] and c["y1"] <= click_y <= c["y2"]:
            return c["track_id"]
    return None


def run_full_analysis(full_tracks, fps, target_ids, H, video_path, output_dir):
    """target_ids: {track_id: "player1" 또는 "player2"}"""
    os.makedirs(output_dir, exist_ok=True)

    df = full_tracks[full_tracks["track_id"].isin(target_ids.keys())].copy()
    if df.empty:
        raise ValueError("선택한 선수를 추적하지 못했습니다.")

    df["player"] = df["track_id"].map(target_ids)
    df["pixel_x"] = ((df["x1"] + df["x2"]) / 2).astype(int)
    df["pixel_y"] = df["y2"]

    smoothed = []
    for player, group in df.groupby("player"):
        group = group.sort_values("frame").reset_index(drop=True)
        group["pixel_x_smooth"] = group["pixel_x"].rolling(window=9, center=True, min_periods=1).mean()
        group["pixel_y_smooth"] = group["pixel_y"].rolling(window=9, center=True, min_periods=1).mean()
        smoothed.append(group)
    df = pd.concat(smoothed, ignore_index=True)

    pixel_coords = df[["pixel_x_smooth", "pixel_y_smooth"]].values.astype(np.float32).reshape(-1, 1, 2)
    court_coords = cv2.perspectiveTransform(pixel_coords, H).reshape(-1, 2)
    df["court_x"] = court_coords[:, 0]
    df["court_y"] = court_coords[:, 1]

    results_list = []
    stats = {}
    for player, group in df.groupby("player"):
        group = group.sort_values("frame").reset_index(drop=True)
        group["court_x_smooth"] = group["court_x"].rolling(window=9, center=True, min_periods=1).mean()
        group["court_y_smooth"] = group["court_y"].rolling(window=9, center=True, min_periods=1).mean()

        dx = group["court_x_smooth"].diff()
        dy = group["court_y_smooth"].diff()
        dt = group["time"].diff()
        group["distance"] = np.sqrt(dx**2 + dy**2)
        group["speed"] = group["distance"] / dt

        stats[player] = {
            "total_distance_m": round(float(group["distance"].sum()), 2),
            "avg_speed_mps": round(float(group["speed"].mean()), 2),
            "max_speed_mps": round(float(group["speed"].max()), 2),
        }
        results_list.append(group)

    final_df = pd.concat(results_list, ignore_index=True)
    csv_path = os.path.join(output_dir, "player_tracking_analysis.csv")
    final_df.to_csv(csv_path, index=False)

    trajectory_path = _draw_trajectory(video_path, final_df, output_dir)
    heatmap_path = _draw_heatmap(final_df, output_dir)

    return {
        "csv_path": csv_path,
        "trajectory_image": trajectory_path,
        "heatmap_image": heatmap_path,
        "stats": stats,
    }


def _draw_trajectory(video_path, df, output_dir):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    player_colors = {
        "player1": ((255, 0, 0), (0, 0, 255)),
        "player2": ((0, 255, 255), (255, 0, 255)),
    }

    for player in sorted(df["player"].unique()):
        sub = df[df["player"] == player].reset_index(drop=True)
        points = sub[["pixel_x", "pixel_y"]].values.astype(int)
        n = len(points)
        start_color, end_color = player_colors.get(player, ((255, 255, 255), (255, 255, 255)))
        for i in range(1, n):
            t = i / (n - 1)
            color = tuple(int(start_color[c] * (1 - t) + end_color[c] * t) for c in range(3))
            cv2.line(frame, tuple(points[i - 1]), tuple(points[i]), color, 2)

    path = os.path.join(output_dir, "trajectory.jpg")
    cv2.imwrite(path, frame)
    return path


def _draw_heatmap(df, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BG, FG, GRID = "#141c19", "#eef2ec", "#8a9891"
    BIN = 0.25                              # 한 칸 = 25cm x 25cm (가로세로 같은 크기)
    X_LO, X_HI = -1.0, COURT_W + 1.0        # 가로축은 코트 폭 전체 + 여백으로 고정

    players = sorted(df["player"].unique())
    fig, axes = plt.subplots(len(players), 1, figsize=(9, 4.2 * len(players)), squeeze=False)
    fig.patch.set_facecolor(BG)

    for ax, player in zip(axes[:, 0], players):
        sub = df[df["player"] == player].dropna(subset=["court_x_smooth", "court_y_smooth"])
        y_lo = sub["court_y_smooth"].min() - 1.0
        y_hi = sub["court_y_smooth"].max() + 1.0
        if y_hi - y_lo < 5.0:               # 너무 납작해지지 않게 최소 세로 범위 확보
            y_hi = y_lo + 5.0

        xs = np.arange(X_LO, X_HI + BIN, BIN)
        ys = np.arange(y_lo, y_hi + BIN, BIN)
        heat, _, _ = np.histogram2d(sub["court_x_smooth"], sub["court_y_smooth"], bins=[xs, ys])
        heat = np.ma.masked_where(heat == 0, heat)

        ax.set_facecolor("black")
        ax.imshow(heat.T, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]],
                  cmap="hot", aspect="equal", interpolation="nearest")   # aspect equal: 1m = 1m

        # 코트 선 (범위 안에 있는 것만): 베이스라인, 서비스라인, 네트, 사이드라인
        for y, name, lw in [(0, "baseline", 2.5), (5.485, "service line", 1.5), (11.885, "net", 2.0),
                            (18.285, "service line", 1.5), (COURT_L, "baseline", 2.5)]:
            if y_lo <= y <= y_hi:
                ax.plot([0, COURT_W], [y, y], color="white", lw=lw, alpha=0.85)
                ax.text(COURT_W + 0.1, y, name, color=GRID, fontsize=8, va="center")
        for x in (0, COURT_W):
            ax.plot([x, x], [max(y_lo, 0), min(y_hi, COURT_L)], color="white", lw=1.5, alpha=0.85)

        ax.set_xlim(X_LO, X_HI)
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(f"{player} Heatmap", color=FG)
        ax.set_xlabel("Court width (m)", color=FG)
        ax.set_ylabel("Distance from baseline (m)", color=FG)
        ax.tick_params(colors=FG)
        for spine in ax.spines.values():
            spine.set_color(GRID)

    plt.tight_layout()
    path = os.path.join(output_dir, "heatmap.png")
    plt.savefig(path, facecolor=BG, dpi=110)
    plt.close(fig)
    return path