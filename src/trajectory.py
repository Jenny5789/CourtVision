import cv2
import pandas as pd

df = pd.read_csv("results/player_tracking.csv")

video_path = "data/raw/archive/video1.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("영상을 읽을 수 없습니다.")
    exit()

player_colors = {
    "player1": ((255, 0, 0), (0, 0, 255)),      # 파랑 -> 빨강
    "player2": ((0, 255, 255), (255, 0, 255)),  # 노랑 -> 자홍
}

for player in sorted(df["player"].unique()):
    sub = df[df["player"] == player].reset_index(drop=True)
    points = sub[["pixel_x", "pixel_y"]].values.astype(int)
    n = len(points)
    start_color, end_color = player_colors[player]

    for i in range(1, n):
        t = i / (n - 1)
        color = tuple(int(start_color[c] * (1 - t) + end_color[c] * t) for c in range(3))
        cv2.line(frame, tuple(points[i - 1]), tuple(points[i]), color, 2)

display_width = 960
h, w = frame.shape[:2]
resized = cv2.resize(frame, (display_width, int(h * display_width / w)))

cv2.imshow("Player Trajectory", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()