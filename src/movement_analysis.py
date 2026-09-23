import pandas as pd
import numpy as np

df = pd.read_csv("results/player_tracking_court.csv")

results = []

for player, group in df.groupby("player"):
    group = group.sort_values("frame").reset_index(drop=True)

    group["court_x_smooth"] = group["court_x"].rolling(window=9, center=True, min_periods=1).mean()
    group["court_y_smooth"] = group["court_y"].rolling(window=9, center=True, min_periods=1).mean()

    dx = group["court_x_smooth"].diff()
    dy = group["court_y_smooth"].diff()
    dt = group["time"].diff()

    group["distance"] = np.sqrt(dx**2 + dy**2)
    group["speed"] = group["distance"] / dt

    total_distance = group["distance"].sum()
    avg_speed = group["speed"].mean()
    max_speed = group["speed"].max()

    print(f"[{player}] 총 이동거리: {total_distance:.2f} m, 평균 속도: {avg_speed:.2f} m/s, 최고 속도: {max_speed:.2f} m/s")

    results.append(group)

final_df = pd.concat(results, ignore_index=True)
final_df.to_csv("results/player_tracking_analysis.csv", index=False)