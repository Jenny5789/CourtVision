import pandas as pd

df = pd.read_csv("results/player_tracking_court.csv")

for player, group in df.groupby("player"):
    print(f"\n[{player}] 행 개수: {len(group)}")
    print(f"court_x 범위: {group['court_x'].min():.2f} ~ {group['court_x'].max():.2f}")
    print(f"court_y 범위: {group['court_y'].min():.2f} ~ {group['court_y'].max():.2f}")