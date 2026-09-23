import pandas as pd

df = pd.read_csv("results/player_tracking_analysis.csv")

max_idx = df["speed"].idxmax()
print("최고 속도 발생 지점 주변:")
print(df.loc[max_idx - 3 : max_idx + 3][["frame", "time", "pixel_x", "pixel_y", "court_x", "court_y", "distance", "speed"]])