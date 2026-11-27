"""선수 추적 안정성 비교 (정답 위치 사용): 정답 선수 한 명을 추적 번호(track id) 하나로 얼마나 끊김 없이 따라가는가.

서비스는 선수 한 명을 track id 하나로만 따라가므로, id 가 바뀌면 이후 이동거리·속도가 사라진다.
정답: data/raw/archive/{video}_player.csv (player_l, player_r). 화면 밖 좌표(x>1900, y<20)는 "선수 없음".
지표(정답 선수별, 박스+10% 여유 안에 정답 점이 들어오는 박스의 track id 기준)
- 적중 프레임 비율, 사용된 서로 다른 id 수, 가장 많이 쓰인 id 의 비율(= 서비스처럼 id 하나만 따라갈 때 얻는 최대 커버리지),
  id 가 바뀐 횟수
사용: python src/eval_tracking_stability.py [tracker.yaml] [conf]
"""
import os
import sys
from collections import Counter

import cv2
import pandas as pd
from ultralytics import YOLO

MARGIN, IMGSZ = 0.10, 960
VIDEOS = ["video9", "video10"]
MODELS = [("yolo11n", "data/runtime_bench/yolo11n.pt"), ("yolo26n", "data/runtime_bench/yolo26n.pt")]


def inside(box, pt):
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * MARGIN, (y2 - y1) * MARGIN
    return x1 - mx <= pt[0] <= x2 + mx and y1 - my <= pt[1] <= y2 + my


def gt_points(row):
    out = {}
    for side, x, y in (("l", row.player_l_x, row.player_l_y), ("r", row.player_r_x, row.player_r_y)):
        if not (x > 1900 and y < 20):
            out[side] = (x, y)
    return out


def evaluate(model, video, tracker, conf):
    gt = pd.read_csv(f"data/raw/archive/{video}_player.csv")
    cap = cv2.VideoCapture(f"data/raw/archive/{video}.mp4")
    seqs = {"l": [], "r": []}          # 프레임별 id (없으면 None), 정답이 있는 프레임만
    for row in gt.itertuples():
        ok, frame = cap.read()
        if not ok:
            break
        res = model.track(frame, classes=[0], conf=conf, imgsz=IMGSZ, tracker=tracker, persist=True, verbose=False)[0]
        boxes = [(int(b.id[0]) if b.id is not None else -1, [float(v) for v in b.xyxy[0]]) for b in res.boxes]
        for side, p in gt_points(row).items():
            hit = [tid for tid, bb in boxes if inside(bb, p)]
            seqs[side].append(hit[0] if hit else None)
    cap.release()
    out = {}
    for side, seq in seqs.items():
        n = len(seq)
        if n == 0:
            continue
        ids = [s for s in seq if s is not None]
        cnt = Counter(ids)
        switches = sum(1 for a, b in zip(ids, ids[1:]) if a != b)
        out[side] = {"정답 프레임": n, "적중 %": round(len(ids) / n * 100, 1), "id 종류": len(cnt),
                     "최다 id 커버리지 %": round(max(cnt.values()) / n * 100, 1) if cnt else 0.0, "id 바뀐 횟수": switches}
    return out


def main():
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    tracker = sys.argv[1] if len(sys.argv) > 1 else "botsort.yaml"
    conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
    lines = [f"tracker {tracker}, conf {conf}, imgsz {IMGSZ}"]
    for name, path in MODELS:
        for video in VIDEOS:
            r = evaluate(YOLO(path), video, tracker, conf)
            for side, v in r.items():
                line = f"{name} {video} {side}: {v}"
                print(line, flush=True)
                lines.append(line)
    with open("results/tracking_stability_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
