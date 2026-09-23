import pandas as pd

df = pd.read_csv("results/player_tracking_court.csv")

for player, group in df.groupby("player"):
    group = group.sort_values("frame").reset_index(drop=True)
    jump = group["court_y"].diff().abs()
    big_jumps = group[jump > 5]  # 한 프레임 사이에 5m 이상 튄 지점
    print(f"\n[{player}] 큰 점프 발생 지점 수: {len(big_jumps)}")
    if len(big_jumps) > 0:
        print(big_jumps[["frame", "time", "pixel_x", "pixel_y", "court_y"]].head(10))