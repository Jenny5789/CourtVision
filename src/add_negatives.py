import cv2
import pandas as pd
import os

ARCHIVE_DIR = "data/raw/archive"
IMG_DIR = "data/ball_dataset/images"
LABEL_DIR = "data/ball_dataset/labels"
STRIDE = 6       # 조건에 맞는 프레임 중 6개당 1개만 사용
MARGIN = 8       # 앞뒤 8프레임도 모두 공이 없어야 함

for vid in range(1, 11):
    video_path = os.path.join(ARCHIVE_DIR, f"video{vid}.mp4")
    csv_path = os.path.join(ARCHIVE_DIR, f"video{vid}_ball.csv")
    if not (os.path.exists(video_path) and os.path.exists(csv_path)):
        continue

    df = pd.read_csv(csv_path)
    absent = [bool(1900 < x < 1920 and y < 20) for x, y in zip(df["ball_x"], df["ball_y"])]
    n = len(absent)

    cap = cv2.VideoCapture(video_path)
    saved = 0
    picked = 0
    for i in range(n):
        ret, frame = cap.read()
        if not ret:
            break
        lo, hi = max(0, i - MARGIN), min(n, i + MARGIN + 1)
        if all(absent[lo:hi]):
            picked += 1
            if picked % STRIDE == 0:
                name = f"video{vid}_neg_{i:04d}"
                cv2.imwrite(os.path.join(IMG_DIR, name + ".jpg"), frame)
                open(os.path.join(LABEL_DIR, name + ".txt"), "w").close()   # 빈 라벨 = 배경
                saved += 1
    cap.release()
    print(f"video{vid}: 배경 {saved}장 추가")