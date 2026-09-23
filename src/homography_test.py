import numpy as np
import cv2

# Pixel 좌표 (video1_court.csv, frame_000 기준) - BL, BR, TR, TL
pixel_points = np.array([
    [64, 872],
    [1806, 865],
    [1186, 601],
    [692, 598],
], dtype=np.float32)

# 실제 코트 좌표 (미터 단위) - 복식 코트 기준, BL을 원점(0,0)으로
# 폭 10.97m, 길이 23.77m
court_points = np.array([
    [0, 0],
    [10.97, 0],
    [10.97, 23.77],
    [0, 23.77],
], dtype=np.float32)

H, _ = cv2.findHomography(pixel_points, court_points)
print("Homography 행렬:")
print(H)

# 테스트: player_tracking.csv의 첫 좌표를 변환해보기
import pandas as pd
df = pd.read_csv("results/player_tracking.csv")
test_pixel = np.array([[df.iloc[0]["pixel_x"], df.iloc[0]["pixel_y"]]], dtype=np.float32)
test_pixel = np.array([test_pixel])

court_coord = cv2.perspectiveTransform(test_pixel, H)
print(f"Pixel 좌표: {test_pixel[0][0]}")
print(f"변환된 Court 좌표(m): {court_coord[0][0]}")