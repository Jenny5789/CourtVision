import sys
import cv2
import math
import pandas as pd
from ultralytics import YOLO

VIDEO = "video9"
MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "models/tennis_ball_best.pt"
model = YOLO(MODEL_PATH)
print("모델:", MODEL_PATH)

gt = pd.read_csv(f"data/raw/archive/{VIDEO}_ball.csv").set_index("frame")

records = []
cap = cv2.VideoCapture(f"data/raw/archive/{VIDEO}.mp4")
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
    boxes = model.predict(frame, conf=0.1, imgsz=960, verbose=False)[0].boxes
    preds = [(float(c), float(b[0]), float(b[1])) for b, c in zip(boxes.xywh, boxes.conf)]
    records.append((has_gt, x, y, preds))
cap.release()

print("conf   적중률(공 있는 프레임)   오검출률(공 없는 프레임)")
for thr in [0.25, 0.35, 0.45, 0.55, 0.65]:
    hit = n_gt = fp = n_no = 0
    for has_gt, x, y, preds in records:
        ps = [p for p in preds if p[0] >= thr]
        if has_gt:
            n_gt += 1
            if ps:
                best = max(ps)
                if math.hypot(best[1] - x, best[2] - y) <= 20:
                    hit += 1
        else:
            n_no += 1
            if ps:
                fp += 1
    print(f"{thr:.2f}   {hit / n_gt * 100:5.1f}%                  {fp / n_no * 100:5.1f}%")