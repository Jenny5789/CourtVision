import cv2
import pandas as pd
import os

ARCHIVE_DIR = "data/raw/archive"
OUTPUT_DIR = "data/ball_dataset"
IMG_DIR = os.path.join(OUTPUT_DIR, "images")
LABEL_DIR = os.path.join(OUTPUT_DIR, "labels")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

# 공의 실제 지름(m) - 원근 보정용 (나중에 정밀화 가능, 우선 고정 픽셀 크기로 시작)
BOX_SIZE = 16  # 임시: 공 주변 16x16px 박스 (고정 크기, 우선 단순하게 시작)

video_ids = range(1, 11)

for vid in video_ids:
    video_path = os.path.join(ARCHIVE_DIR, f"video{vid}.mp4")
    ball_csv_path = os.path.join(ARCHIVE_DIR, f"video{vid}_ball.csv")

    if not os.path.exists(video_path) or not os.path.exists(ball_csv_path):
        print(f"video{vid} 파일 없음, 스킵")
        continue

    ball_df = pd.read_csv(ball_csv_path)
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_name = f"frame_{frame_idx:03d}"
        row = ball_df[ball_df["frame"] == frame_name]

        if not row.empty and frame_idx % 2 == 0:  # 프레임 절반만 사용 (용량/시간 절약)
            x, y = row.iloc[0]["ball_x"], row.iloc[0]["ball_y"]

            # 화면 밖 sentinel 값(공 안 보임)은 제외
            if not (1900 < x < 1920 and y < 20):
                img_name = f"video{vid}_{frame_idx:04d}.jpg"
                cv2.imwrite(os.path.join(IMG_DIR, img_name), frame)

                # YOLO 라벨 형식: class_id center_x center_y width height (모두 0~1 정규화)
                cx, cy = x / width, y / height
                bw, bh = BOX_SIZE / width, BOX_SIZE / height

                label_path = os.path.join(LABEL_DIR, img_name.replace(".jpg", ".txt"))
                with open(label_path, "w") as f:
                    f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

                saved += 1

        frame_idx += 1

    cap.release()
    print(f"video{vid}: {saved}장 저장")

print("데이터셋 준비 완료:", OUTPUT_DIR)