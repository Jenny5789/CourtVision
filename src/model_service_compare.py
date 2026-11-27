"""서비스 파이프라인 기준 모델 비교: yolo11n vs yolo26n.

같은 영상·같은 코트 보정으로 선수 검출·추적 -> 이동거리/속도까지 전부 돌려서
서비스 지표(총 이동거리, 평균/최고 속도)가 모델에 따라 얼마나 달라지는지 비교한다.
(int8_service_compare.py 와 같은 방식, 비교 대상만 다름)

주의:
- 코트 4점은 int8_service_compare.py 와 같은 눈대중 근사 좌표. 두 모델에 동일하게 적용하므로
  절대 정확도가 아니라 모델 간 상대 비교에 의미가 있다.
- 영상 1개(video9), 1회 실행. 실제 이동거리 정답은 없음.
- 두 모델 모두 PyTorch 기본 장치(GPU 가능 시 GPU)로 실행(서비스와 같은 방식).
사용: python src/model_service_compare.py
"""
import os
import sys
import time

import cv2
import pandas as pd
from ultralytics import YOLO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "webapp", "backend"))
import pipeline  # compute_homography, court_polygon_pixels, get_candidates_at_time, run_full_analysis

VIDEO = "data/raw/archive/video9.mp4"
IMGSZ = 960
CONF = 0.3
CORNERS = [[690, 610], [1230, 610], [1130, 435], [795, 435]]
MODELS = [("yolo11n", "data/runtime_bench/yolo11n.pt"), ("yolo26n", "data/runtime_bench/yolo26n.pt")]


def track_full_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    rows = []
    frame_idx = 0
    t0 = time.perf_counter()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.track(frame, classes=[0], conf=CONF, imgsz=IMGSZ, tracker="botsort.yaml",
                              persist=True, verbose=False)
        for box in results[0].boxes:
            tid = int(box.id[0]) if box.id is not None else -1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            rows.append({"frame": frame_idx, "time": round(frame_idx / fps, 3),
                         "track_id": tid, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        frame_idx += 1
    cap.release()
    return pd.DataFrame(rows), fps, time.perf_counter() - t0, frame_idx


def main():
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    H = pipeline.compute_homography(CORNERS)
    court_polygon = pipeline.court_polygon_pixels(H)

    all_stats, extra = {}, {}
    for name, path in MODELS:
        print(f"\n=== {name} ===", flush=True)
        model = YOLO(path)
        tracks, fps, elapsed, n_frames = track_full_video(VIDEO, model)
        print(f"프레임 {n_frames}개, {elapsed:.1f}초 ({elapsed / n_frames * 1000:.0f}ms/프레임), fps={fps:.1f}", flush=True)

        # 이전 방식(첫 프레임에서 코트 근처 가장 큰 박스)은 video9 첫 프레임에서 먼 쪽 선수(서브하는 사람)를 고르므로 참고로만 기록
        _, auto_id = pipeline.get_candidates_at_time(tracks, fps, court_polygon, 0.0)

        # 정답 위치(player_l, player_r)와 겹치는 박스의 track id 중 가장 많이 나온 id 를 선수 id 로 사용
        gt = pd.read_csv("data/raw/archive/video9_player.csv")
        by_frame = {f: g for f, g in tracks.groupby("frame")}
        votes = {"l": {}, "r": {}}
        for row in gt.itertuples():
            g = by_frame.get(row.Index)
            if g is None:
                continue
            for side, (x, y) in (("l", (row.player_l_x, row.player_l_y)), ("r", (row.player_r_x, row.player_r_y))):
                if x > 1900 and y < 20:
                    continue
                for b in g.itertuples():
                    mx, my = (b.x2 - b.x1) * 0.1, (b.y2 - b.y1) * 0.1
                    if b.x1 - mx <= x <= b.x2 + mx and b.y1 - my <= y <= b.y2 + my:
                        votes[side][b.track_id] = votes[side].get(b.track_id, 0) + 1
                        break
        ids = {s: max(v, key=v.get) for s, v in votes.items() if v}
        print(f"정답 기준 선수 id: {ids} (이전 방식 자동 선택 id: {auto_id})", flush=True)
        targets = {}
        if "l" in ids:
            targets[ids["l"]] = "player1"
        if "r" in ids and ids.get("r") != ids.get("l"):
            targets[ids["r"]] = "player2"

        result = pipeline.run_full_analysis(tracks, fps, targets, H, VIDEO, f"results/model_compare_{name}")
        print("stats:", result["stats"], flush=True)
        all_stats[name] = {"stats": result["stats"], "ms_per_frame": round(elapsed / n_frames * 1000, 1)}
        extra[name] = {"전체 트랙 수": int(tracks["track_id"].nunique()), "정답 기준 선수 id": ids,
                       "이전 방식 자동 선택 id": auto_id, "전체 프레임": n_frames}

    print("\n\n=== 최종 비교 (정답 기준 선수) ===")
    names = list(all_stats)
    for n in names:
        print(n, all_stats[n], extra[n])

    with open("results/model_service_compare_log.txt", "w", encoding="utf-8") as f:
        f.write(f"영상: {VIDEO}\n코트 4점(근사): {CORNERS}\nconf {CONF}, imgsz {IMGSZ}, BoT-SORT 기본\n")
        f.write("선수 id 는 정답 위치(player_l·player_r = 카메라에서 가까운 선수 한 명의 좌·우 발)와 겹치는 박스의 최다 track id\n")
        f.write("참고: 이전 방식(첫 프레임 코트 근처 최대 박스)은 video9 첫 프레임에서 먼 쪽 선수(서브하는 사람, 추적 번호 2)를 골랐음 -> 11n 33.83m / 26n 0.17m 는 먼 쪽 선수 기준값이며, 26n에서 이 선수의 추적 번호가 쪼개져 일부만 계산된 실제 차이(results/model_selection_experiment.md 참고)\n\n")
        for n in names:
            f.write(f"{n}: {all_stats[n]} | {extra[n]}\n")


if __name__ == "__main__":
    main()
