import cv2

video_path = "data/raw/archive/video1.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("영상을 열 수 없습니다.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"FPS: {fps}, Width: {width}, Height: {height}, Frame Count: {frame_count}")

display_width = 960
display_height = int(height * display_width / width)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized = cv2.resize(frame, (display_width, display_height))
    cv2.imshow("Tennis Video", resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()