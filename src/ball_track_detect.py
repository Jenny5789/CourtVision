"""공 추적 평가용 검출 후보 캐시.

v2 공 검출 모델로 프레임마다 후보(x, y, conf)를 낮은 임계값(0.1)으로 모두 저장한다.
추적 파라미터를 바꿔 가며 실험할 때 검출을 다시 돌리지 않기 위한 캐시다.

대상
- video9, video10 (data/raw/archive): 공이 없는 프레임이 섞여 있어 오검출 평가가 가능.
  v2 학습에는 video1~8만 사용. video9·10은 best epoch 선택에 쓰였으므로 완전한 시험 데이터는 아님.
- TrackNet 테니스 데이터셋 (data/raw/Dataset): 방송 영상 10경기, 1280x720, 30fps, 95개 클립.
  v2 학습에 쓰지 않은 데이터(처음 보는 장면).

사용: python src/ball_track_detect.py [모델 경로] [출력 pkl]
"""
import glob
import os
import pickle
import sys
import time

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "models/tennis_ball_v2.pt"
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "data/ball_tracking/dets_v2.pkl"
CONF, IMGSZ = 0.1, 960
model = YOLO(MODEL_PATH)


def candidates(result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 3), np.float32)
    xy = boxes.xywh[:, :2].cpu().numpy()
    conf = boxes.conf.cpu().numpy()[:, None]
    return np.hstack([xy, conf]).astype(np.float32)


def archive_sequence(name):
    gt = pd.read_csv(f"data/raw/archive/{name}_ball.csv")
    cap = cv2.VideoCapture(f"data/raw/archive/{name}.mp4")
    fps, width, height = cap.get(cv2.CAP_PROP_FPS), int(cap.get(3)), int(cap.get(4))
    dets = []
    for _ in range(len(gt)):
        ok, frame = cap.read()
        if not ok:
            break
        dets.append(candidates(model.predict(frame, conf=CONF, imgsz=IMGSZ, verbose=False)[0]))
    cap.release()
    visible = ~((gt.ball_x > 1900) & (gt.ball_x < 1920) & (gt.ball_y < 20))
    truth = np.stack([visible.values.astype(np.float32), gt.ball_x.values, gt.ball_y.values], 1)[: len(dets)]
    return {"W": width, "H": height, "fps": fps, "dets": dets, "gt": truth}


def clip_sequence(clip_dir):
    label = pd.read_csv(os.path.join(clip_dir, "Label.csv"))
    paths = [os.path.join(clip_dir, name) for name in label["file name"]]
    dets = []
    for i in range(0, len(paths), 8):          # 한 번에 많이 올리면 GPU 메모리가 부족해 8장씩 처리
        dets.extend(candidates(r) for r in model.predict(paths[i:i + 8], conf=CONF, imgsz=IMGSZ, verbose=False))
    visible = (label["visibility"] > 0).values.astype(np.float32)
    truth = np.stack([visible, label["x-coordinate"].values, label["y-coordinate"].values], 1).astype(np.float32)
    return {"W": 1280, "H": 720, "fps": 30.0, "dets": dets, "gt": truth}


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    data = {}
    t0 = time.time()
    for name in ["video9", "video10"]:
        data[name] = archive_sequence(name)
        print(f"{name}: {len(data[name]['dets'])}프레임 ({time.time() - t0:.0f}s)", flush=True)
    clips = sorted(glob.glob("data/raw/Dataset/game*/Clip*"),
                   key=lambda p: (int(p.split("game")[1].split(os.sep)[0]), int(p.rsplit("Clip", 1)[1])))
    for i, clip in enumerate(clips, 1):
        key = clip.split("Dataset" + os.sep)[1].replace(os.sep, "/")
        data[key] = clip_sequence(clip)
        if i % 10 == 0 or i == len(clips):
            print(f"클립 {i}/{len(clips)} 완료 ({time.time() - t0:.0f}s)", flush=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump({"model": MODEL_PATH, "conf": CONF, "imgsz": IMGSZ, "data": data}, f)
    print("저장:", OUT_PATH, f"({os.path.getsize(OUT_PATH) / 1e6:.1f}MB)")


if __name__ == "__main__":
    main()
