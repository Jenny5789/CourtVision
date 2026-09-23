"""실험 A: v3의 배경 비율 희석을 보정하기 위해 배경 이미지를 추가로 뽑는다.
add_negatives.py와 같은 조건(앞뒤 8프레임 모두 공 없음)이되, 기존 선택(6개당 1개, picked%6==0)과 겹치지 않게 picked%6==3을 뽑는다.
학습 영상(video1~8)에서만 뽑고, 검증 영상(video9/10)은 건드리지 않는다. 기존 v2 데이터셋 폴더는 수정하지 않고 별도 폴더에 저장한다."""
import cv2
import pandas as pd
import os

ARCHIVE_DIR = "data/raw/archive"
OUT = "data/ball_dataset_v3a_neg"
STRIDE, OFFSET, MARGIN = 6, 3, 8
os.makedirs(f"{OUT}/images", exist_ok=True)
os.makedirs(f"{OUT}/labels", exist_ok=True)

total = 0
for vid in range(1, 9):
    df = pd.read_csv(f"{ARCHIVE_DIR}/video{vid}_ball.csv")
    absent = [bool(1900 < x < 1920 and y < 20) for x, y in zip(df["ball_x"], df["ball_y"])]
    n = len(absent)
    cap = cv2.VideoCapture(f"{ARCHIVE_DIR}/video{vid}.mp4")
    picked = saved = 0
    for i in range(n):
        ret, frame = cap.read()
        if not ret:
            break
        lo, hi = max(0, i - MARGIN), min(n, i + MARGIN + 1)
        if all(absent[lo:hi]):
            picked += 1
            if picked % STRIDE == OFFSET:
                name = f"video{vid}_negB_{i:04d}"
                cv2.imwrite(f"{OUT}/images/{name}.jpg", frame)
                open(f"{OUT}/labels/{name}.txt", "w").close()
                saved += 1
    cap.release()
    total += saved
    print(f"video{vid}: 배경 {saved}장 추가")
print("합계", total)
