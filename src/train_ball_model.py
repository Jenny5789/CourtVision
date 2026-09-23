from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")

    model.train(
        data="data/ball_dataset/data.yaml",
        epochs=50,
        imgsz=960,
        batch=16,
        device=0,
        patience=15,
        project="runs/ball_detection",
        name="v2",
    )