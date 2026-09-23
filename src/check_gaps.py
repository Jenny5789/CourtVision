import pandas as pd

df = pd.read_csv("results/player_tracking.csv")
all_frames = set(range(689))
tracked_frames = set(df["frame"])
missing = sorted(all_frames - tracked_frames)
print(f"빠진 프레임: {missing}")
