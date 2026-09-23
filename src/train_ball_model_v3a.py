from ultralytics import YOLO

# v3a = v3 + 배경 491장 추가(배경 비율 보정 실험). 검증은 video9/10 그대로(변경 없음).
if __name__ == "__main__":
    model = YOLO("yolo11n.pt")

    model.train(
        data="data/ball_dataset/data_v3a.yaml",
        epochs=50,
        imgsz=960,
        batch=16,
        device=0,
        patience=15,
        project="runs/ball_detection",
        name="v3a",
    )
