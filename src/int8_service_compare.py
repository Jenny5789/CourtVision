"""서비스 파이프라인 기준 INT8 전후 비교: 같은 영상·같은 코트 보정으로
FP32(PyTorch)와 INT8(OpenVINO) 각각 선수 검출·추적 -> 이동거리/속도까지 전부 돌려서
최종 서비스 지표(총 이동거리, 평균/최고 속도)가 얼마나 달라지는지 비교한다.

주의:
- 코트 4점은 실제 클릭이 아니라 첫 프레임을 보고 눈대중으로 잡은 좌표(대략적인 근사치).
  두 모델에 "동일하게" 적용하므로 절대 정확도보다 FP32-INT8 상대 비교에 의미가 있다.
- 측정 1회, 영상 1개.
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
# 첫 프레임에서 눈대중으로 잡은 코트 4점 (서비스라인 좌/우 끝, 네트 아래 좌/우) - 순서 무관(자동 정렬됨)
CORNERS = [[690, 610], [1230, 610], [1130, 435], [795, 435]]

CONFIGS = [
    # (이름, 모델 경로, device, imgsz) - INT8 OpenVINO 모델은 544x960 고정 크기로 내보내져 있어
    # 그 크기를 그대로 줘야 함(안 그러면 셰이프 불일치 오류)
    ("FP32 (PyTorch)", "yolo11n.pt", None, IMGSZ),
    ("INT8 (OpenVINO)", "models/int8/yolo11n_int8_openvino_model", "intel:cpu", [544, 960]),
]


def track_full_video(video_path, model, device, imgsz):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    rows = []
    frame_idx = 0
    t0 = time.perf_counter()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        kwargs = dict(classes=[0], conf=CONF, imgsz=imgsz, tracker="botsort.yaml", persist=True, verbose=False)
        if device is not None:
            kwargs["device"] = device
        results = model.track(frame, **kwargs)
        for box in results[0].boxes:
            tid = int(box.id[0]) if box.id is not None else -1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            rows.append({"frame": frame_idx, "time": round(frame_idx / fps, 3),
                         "track_id": tid, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        frame_idx += 1
    cap.release()
    elapsed = time.perf_counter() - t0
    return pd.DataFrame(rows), fps, elapsed, frame_idx


def main():
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    H = pipeline.compute_homography(CORNERS)
    court_polygon = pipeline.court_polygon_pixels(H)

    all_stats = {}
    for name, model_path, device, imgsz in CONFIGS:
        print(f"\n=== {name} ===")
        model = YOLO(model_path, task="detect")
        full_tracks, fps, elapsed, n_frames = track_full_video(VIDEO, model, device, imgsz)
        print(f"프레임 {n_frames}개, {elapsed:.1f}초 ({elapsed/n_frames*1000:.0f}ms/프레임), fps={fps:.1f}")

        candidates, player1_id = pipeline.get_candidates_at_time(full_tracks, fps, court_polygon, 0.0)
        if player1_id is None:
            print("player1을 찾지 못함 - 건너뜀")
            continue
        print(f"player1_id = {player1_id} (후보 {len(candidates)}명)")

        out_dir = f"results/int8_compare_{name.split()[0].lower()}"
        result = pipeline.run_full_analysis(full_tracks, fps, {player1_id: "player1"}, H, VIDEO, out_dir)
        stats = result["stats"]["player1"]
        print("stats:", stats)
        all_stats[name] = {**stats, "elapsed_sec": round(elapsed, 1), "ms_per_frame": round(elapsed / n_frames * 1000, 1)}

    print("\n\n=== 최종 비교 (player1) ===")
    header = ["항목"] + list(all_stats.keys())
    print(" | ".join(header))
    for key in ["total_distance_m", "avg_speed_mps", "max_speed_mps", "ms_per_frame"]:
        row = [key] + [str(all_stats[n][key]) for n in all_stats]
        print(" | ".join(row))

    # 파일로도 저장
    with open("results/int8_service_compare_log.txt", "w", encoding="utf-8") as f:
        f.write(f"영상: {VIDEO}\n코트 4점(근사): {CORNERS}\n\n")
        for n, s in all_stats.items():
            f.write(f"{n}: {s}\n")


if __name__ == "__main__":
    main()
