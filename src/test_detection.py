import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

video_path = "data/raw/archive/video1.mp4"
cap = cv2.VideoCapture(video_path)

display_width = 960  # 화면 표시용 너비 (YOLO모델 내부 구조상 32배의 배수를 요구함)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, classes=[0], conf=0.3, device=0, imgsz=960, verbose=False)
    annotated_frame = results[0].plot()

    # 화면 표시 크기만 축소 (실제 검출은 원본 frame 기준)
    h, w = annotated_frame.shape[:2]
    display_height = int(h * display_width / w)
    resized = cv2.resize(annotated_frame, (display_width, display_height))

    cv2.imshow("Player Detection", resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()