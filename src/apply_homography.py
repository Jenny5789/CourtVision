import numpy as np
import cv2
import pandas as pd

pixel_points = np.array([
    [64, 872],
    [1806, 865],
    [1186, 601],
    [692, 598],
], dtype=np.float32)

court_points = np.array([
    [0, 0],
    [10.97, 0],
    [10.97, 23.77],
    [0, 23.77],
], dtype=np.float32)

H, _ = cv2.findHomography(pixel_points, court_points)

df = pd.read_csv("results/player_tracking.csv")

# Pixel 단계에서 먼저 스무딩 (Homography로 증폭되기 전에 잡음 제거)
smoothed = []
for player, group in df.groupby("player"):
    group = group.sort_values("frame").reset_index(drop=True)
    group["pixel_x_smooth"] = group["pixel_x"].rolling(window=9, center=True, min_periods=1).mean()
    group["pixel_y_smooth"] = group["pixel_y"].rolling(window=9, center=True, min_periods=1).mean()
    smoothed.append(group)

df = pd.concat(smoothed, ignore_index=True)

pixel_coords = df[["pixel_x_smooth", "pixel_y_smooth"]].values.astype(np.float32).reshape(-1, 1, 2)
court_coords = cv2.perspectiveTransform(pixel_coords, H).reshape(-1, 2)

df["court_x"] = court_coords[:, 0]
df["court_y"] = court_coords[:, 1]

df.to_csv("results/player_tracking_court.csv", index=False)

for player, group in df.groupby("player"):
    print(f"[{player}] court_x: {group['court_x'].min():.2f}~{group['court_x'].max():.2f}, "
          f"court_y: {group['court_y'].min():.2f}~{group['court_y'].max():.2f}")