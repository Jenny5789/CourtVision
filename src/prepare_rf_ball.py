"""Roboflow 'Tennis Ball and Court Detection'(CC BY 4.0) 정리.

- ball(0번)만 사용, 나머지 클래스 라벨은 제거
- 증강 복사본 제거: 파일명의 '.rf.' 앞부분이 같으면 같은 원본 -> 세 split 전체에서 1장만 남김
- 공이 하나라도 MAX_SIDE_PX 이상인 이미지는 제외(근접 사진; 큰 공만 지우면 라벨 누락이 생기므로 이미지째 제외)
- 공이 없는 이미지도 제외
"""
import glob
import os
import shutil
from collections import Counter

SRC = "data/external/roboflow_tennis"
DST = "data/external/rf_ball_small"
MAX_SIDE_PX = 50     # 640px 기준 공 한 변
IMG = 640

os.makedirs(f"{DST}/images", exist_ok=True)
os.makedirs(f"{DST}/labels", exist_ok=True)

stem = lambda p: os.path.basename(p).split(".rf.")[0]
seen = set()
stats = Counter()

for split in ["train", "valid", "test"]:
    for img in sorted(glob.glob(f"{SRC}/{split}/images/*")):
        st = stem(img)
        if st in seen:
            stats["duplicate_skipped"] += 1
            continue
        seen.add(st)
        lab = f"{SRC}/{split}/labels/{os.path.basename(img).rsplit('.', 1)[0]}.txt"
        balls = [l.split() for l in open(lab).read().splitlines() if l.strip() and l.split()[0] == "0"]
        if not balls:
            stats["no_ball"] += 1
            continue
        if max(max(float(b[3]), float(b[4])) * IMG for b in balls) >= MAX_SIDE_PX:
            stats["large_ball"] += 1
            continue
        ext = img.rsplit(".", 1)[1]
        shutil.copy(img, f"{DST}/images/{st}.{ext}")
        with open(f"{DST}/labels/{st}.txt", "w") as f:
            f.write("\n".join(" ".join(b) for b in balls) + "\n")
        stats["kept_images"] += 1
        stats["kept_balls"] += len(balls)

print(dict(stats))
with open(f"{DST}/LICENSE.txt", "w", encoding="utf-8") as f:
    f.write("Source: https://universe.roboflow.com/new-workspace-em6aw/tennis-ball-and-court-detection\n"
            "License: CC BY 4.0 (attribution required). Filtered: ball only, deduplicated, ball < 50px.\n")
