import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

df = pd.read_csv("results/player_tracking_analysis.csv")

# ===== 조정 가능한 설정값 =====
bin_size = 0.5   # 한 칸의 실제 크기(m) — 작을수록 세밀함
x_range = (0, 13)
y_range = (-5, 65)

fig, ax = plt.subplots(figsize=(5, 7))

cmaps = {"player1": "Reds", "player2": "Blues"}

x_bins = int((x_range[1] - x_range[0]) / bin_size)
y_bins = int((y_range[1] - y_range[0]) / bin_size)

for player in sorted(df["player"].unique()):
    sub = df[df["player"] == player].dropna(subset=["court_x_smooth", "court_y_smooth"])

    heatmap, xedges, yedges = np.histogram2d(
        sub["court_x_smooth"], sub["court_y_smooth"],
        bins=[x_bins, y_bins],
        range=[x_range, y_range]
    )

    heatmap_masked = np.ma.masked_where(heatmap == 0, heatmap)

    ax.imshow(heatmap_masked.T, origin="lower", extent=[*x_range, *y_range],
               cmap=cmaps.get(player, "Greens"), alpha=0.7, aspect="auto")

ax.set_title("Player1 vs Player2 Heatmap (combined)")
ax.set_xlabel("Court X (m)")
ax.set_ylabel("Court Y (m)")

legend_elements = [
    Patch(facecolor="red", edgecolor="red", alpha=0.7, label="Player1"),
    Patch(facecolor="blue", edgecolor="blue", alpha=0.7, label="Player2"),
]
ax.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.9, -0.15), borderaxespad=0)
plt.tight_layout()
plt.savefig("results/heatmap_combined.png")
plt.show()