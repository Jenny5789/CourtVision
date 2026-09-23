import cv2
import numpy as np
import pandas as pd
import os
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

video_path = "data/raw/archive/video1.mp4"
cap = cv2.VideoCapture(video_path)

court_polygon = np.array([
    [64, 872],
    [1806, 865],
    [1186, 601],
    [692, 598],
], dtype=np.int32)

display_width = 960
target_ids = {}
selected_point = None
labels = ["player1", "player2"]


def on_mouse(event, x, y, flags, param):
    global selected_point
    if event == cv2.EVENT_LBUTTONDOWN:
        scale = param["orig_width"] / display_width
        selected_point = (int(x * scale), int(y * scale))


cv2.namedWindow("Player Tracking")

ret, first_frame = cap.read()
if not ret:
    print("영상을 읽을 수 없습니다.")
    exit()

orig_h, orig_w = first_frame.shape[:2]
cv2.setMouseCallback("Player Tracking", on_mouse, param={"orig_width": orig_w})

# ===== 첫 프레임 Detection/Tracking은 딱 1번만 실행 =====
first_results = model.track(first_frame, classes=[0], conf=0.3, device=0, imgsz=960,
                             tracker="botsort.yaml", persist=True, verbose=False)

candidates = []
for box in first_results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    tid = int(box.id[0]) if box.id is not None else -1
    bottom_center = (int((x1 + x2) / 2), y2)
    if cv2.pointPolygonTest(court_polygon, bottom_center, True) < -150:
        continue
    candidates.append((tid, x1, y1, x2, y2))

# ===== 1. 선수 클릭 선택 (같은 candidates를 계속 재사용, model.track 재호출 없음) =====
while len(target_ids) < 2:
    display_frame = first_frame.copy()
    for tid, x1, y1, x2, y2 in candidates:
        color = (0, 255, 0) if tid not in target_ids else (255, 0, 0)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display_frame, f"ID:{tid}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    resized = cv2.resize(display_frame, (display_width, int(orig_h * display_width / orig_w)))

    msg = f"Click player {len(target_ids) + 1} (or press 'f' to finish with {len(target_ids)})"
    text_pos = (500, 40)
    (text_w, text_h), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(resized, (text_pos[0] - 10, text_pos[1] - text_h - 10),
                  (text_pos[0] + text_w + 10, text_pos[1] + 10), (0, 0, 0), -1)
    cv2.putText(resized, msg, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Player Tracking", resized)
    key = cv2.waitKey(30) & 0xFF

    if key == ord('f') and len(target_ids) >= 1:
        break

    if selected_point is not None:
        for tid, x1, y1, x2, y2 in candidates:
            if x1 <= selected_point[0] <= x2 and y1 <= selected_point[1] <= y2 and tid not in target_ids:
                target_ids[tid] = labels[len(target_ids)]
                print(f"{labels[len(target_ids) - 1]} = Track ID {tid}")
                break
        selected_point = None

print(f"선택 완료: {target_ids}")

# ===== 2. 실제 추적 + CSV 저장 (첫 프레임부터 이어서 처리) =====
fps = cap.get(cv2.CAP_PROP_FPS)
frame_idx = 0
tracking_data = []
pending_frame = first_frame
pending_results = first_results

while True:
    if pending_frame is not None:
        frame = pending_frame
        results = pending_results
        pending_frame = None
    else:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.track(frame, classes=[0], conf=0.3, device=0, imgsz=960,
                               tracker="botsort.yaml", persist=True, verbose=False)

    for box in results[0].boxes:
        tid = int(box.id[0]) if box.id is not None else -1
        if tid not in target_ids:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        pixel_x = int((x1 + x2) / 2)
        pixel_y = y2

        tracking_data.append({
            "frame": frame_idx,
            "time": round(frame_idx / fps, 3),
            "track_id": tid,
            "player": target_ids[tid],
            "pixel_x": pixel_x,
            "pixel_y": pixel_y
        })

        color = (0, 0, 255) if target_ids[tid] == "player1" else (255, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, target_ids[tid], (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    resized = cv2.resize(frame, (display_width, int(orig_h * display_width / orig_w)))
    cv2.imshow("Player Tracking", resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()

os.makedirs("results", exist_ok=True)
df = pd.DataFrame(tracking_data)
df.to_csv("results/player_tracking.csv", index=False)
print(f"저장 완료: results/player_tracking.csv ({len(df)} rows)")