import cv2
import math
import pandas as pd
from ultralytics import YOLO

VIDEO = "video9"
model = YOLO("models/tennis_ball_best.pt")
gt = pd.read_csv(f"data/raw/archive/{VIDEO}_ball.csv").set_index("frame")

cap = cv2.VideoCapture(f"data/raw/archive/{VIDEO}.mp4")
hit = miss = fp = tn = 0
idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    name = f"frame_{idx:03d}"
    idx += 1
    if name not in gt.index:
        continue

    x, y = gt.loc[name, "ball_x"], gt.loc[name, "ball_y"]
    has_gt = not (1900 < x < 1920 and y < 20)

    boxes = model.predict(frame, conf=0.25, imgsz=960, verbose=False)[0].boxes
    if len(boxes) > 0:
        b = boxes.xywh[boxes.conf.argmax()]
        px, py = float(b[0]), float(b[1])
        if has_gt:
            if math.hypot(px - x, py - y) <= 20:
                hit += 1
            else:
                miss += 1
        else:
            fp += 1
    else:
        if has_gt:
            miss += 1
        else:
            tn += 1
cap.release()

print(f"공이 있는 프레임 중 20px 이내로 맞춘 비율: {hit / max(hit + miss, 1) * 100:.1f}%  ({hit}/{hit + miss})")
print(f"공이 없는 프레임에서 오검출한 비율: {fp / max(fp + tn, 1) * 100:.1f}%  ({fp}/{fp + tn})")