import os
import sys

from ultralytics import YOLO

# v4 = v2 학습 조건 그대로 + TrackNet(방송 영상) game1~4 추가. 검증은 video9/10 + TrackNet game5. 시험(game6~10)은 학습·검증에 쓰지 않음.
# 사용: python src/train_ball_model_v4.py [기본 가중치: yolo11n.pt | yolo26n.pt | yolo11s.pt | yolo26s.pt]
# 모델 구조(버전·크기)만 바꿔 같은 데이터·같은 조건으로 학습해 비교한다.
if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "yolo11n.pt"
    model = YOLO(base)

    model.train(
        data="data/ball_dataset/data_v4.yaml",
        epochs=50,
        imgsz=960,
        batch=16,
        device=0,
        patience=15,
        project="runs/ball_detection",
        name="v4_" + os.path.splitext(base)[0],
        exist_ok=True,
    )
