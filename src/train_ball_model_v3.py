from ultralytics import YOLO

# v3 = v2 학습 조건 그대로 + Roboflow(CC BY 4.0) 1,457장 추가. 검증은 video9/10 그대로(변경 없음).
if __name__ == "__main__":
    model = YOLO("yolo11n.pt")

    model.train(
        data="data/ball_dataset/data_v3.yaml",
        epochs=50,
        imgsz=960,
        batch=16,
        device=0,
        patience=15,
        project="runs/ball_detection",
        name="v3",
    )
