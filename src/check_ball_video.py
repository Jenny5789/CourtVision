import cv2
import pandas as pd
from ultralytics import YOLO

VIDEO = "video9"          # 학습에 안 쓴 영상 (video9, video10)
CONF = 0.25
DISPLAY_WIDTH = 1280

model = YOLO("models/tennis_ball_best.pt")
gt = pd.read_csv(f"data/raw/archive/{VIDEO}_ball.csv").set_index("frame")

cap = cv2.VideoCapture(f"data/raw/archive/{VIDEO}.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out_h = int(h * DISPLAY_WIDTH / w)

writer = cv2.VideoWriter(f"results/ball_check_{VIDEO}.mp4",
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (DISPLAY_WIDTH, out_h))

idx = 0
paused = False
while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        name = f"frame_{idx:03d}"
        idx += 1

        if name in gt.index:
            x, y = gt.loc[name, "ball_x"], gt.loc[name, "ball_y"]
            if not (1900 < x < 1920 and y < 20):
                cv2.circle(frame, (int(x), int(y)), 22, (0, 0, 255), 2)     # 정답 = 빨강

        boxes = model.predict(frame, conf=CONF, imgsz=960, verbose=False)[0].boxes
        for b, c in zip(boxes.xywh, boxes.conf):
            px, py = int(b[0]), int(b[1])
            cv2.circle(frame, (px, py), 14, (0, 255, 0), 2)                  # 예측 = 초록
            cv2.putText(frame, f"{float(c):.2f}", (px + 16, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, f"{VIDEO}  frame {idx}   red=정답  green=예측", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        frame = cv2.resize(frame, (DISPLAY_WIDTH, out_h))
        writer.write(frame)
        cv2.imshow("ball check", frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord("q"):
        break
    if key == ord(" "):
        paused = not paused

cap.release()
writer.release()
cv2.destroyAllWindows()
print(f"저장: results/ball_check_{VIDEO}.mp4")