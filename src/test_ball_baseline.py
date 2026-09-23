import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

video_path = "data/raw/archive/video1.mp4"
cap = cv2.VideoCapture(video_path)

detected_count = 0
total_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    total_count += 1

    # COCO 클래스 32번 = sports ball
    results = model.predict(frame, classes=[32], conf=0.1, imgsz=960, verbose=False)
    if len(results[0].boxes) > 0:
        detected_count += 1

cap.release()
print(f"전체 프레임: {total_count}")
print(f"공이 검출된 프레임: {detected_count}")
print(f"검출률: {detected_count / total_count * 100:.1f}%")