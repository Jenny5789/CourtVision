"""TrackNet 테니스 데이터셋을 공 검출 학습용(YOLO 형식)으로 변환해 v4 데이터 목록을 만든다.

- 데이터: data/raw/Dataset (방송 영상 10경기, 1280x720, 30fps, 클립별 Label.csv: visibility, x, y)
- 학습: game1~4 에서 TRAIN_STRIDE 프레임마다 1장   / 검증: game5 에서 VAL_STRIDE 프레임마다 1장
- 시험: game6~10 은 학습·검증에 전혀 쓰지 않는다(평가 전용).
- 공 크기: v2 데이터는 1920x1080 에서 16x16px 고정 박스. 1280x720 에서는 같은 비율인 11px 로 만든다.
- 공이 보이지 않는 프레임(visibility 0)은 빈 라벨(배경 이미지)로 포함한다.
- 기존 v2/v3 파일은 수정하지 않는다. 새 이미지·라벨은 tn_ 접두사로 같은 images/, labels/ 폴더에 추가한다.

출처: TrackNet dataset (broadcast tennis, 10 videos). 학습 데이터로만 사용하며 이미지는 저장소에 올리지 않는다(data/ 는 git 제외).
"""
import glob
import os
import shutil

import pandas as pd

SRC = "data/raw/Dataset"
DST = "data/ball_dataset"
TRAIN_GAMES, VAL_GAMES = [1, 2, 3, 4], [5]
TRAIN_STRIDE, VAL_STRIDE = 3, 6
W, H, BOX = 1280, 720, 11


def convert(games, stride):
    paths, n_ball, n_bg = [], 0, 0
    for game in games:
        clips = sorted(glob.glob(f"{SRC}/game{game}/Clip*"), key=lambda p: int(p.rsplit("Clip", 1)[1]))
        for clip in clips:
            clip_no = int(clip.rsplit("Clip", 1)[1])
            label = pd.read_csv(os.path.join(clip, "Label.csv"))
            for i, row in label.iterrows():
                if i % stride:
                    continue
                name = f"tn_g{game}_c{clip_no}_{i:04d}"
                x, y = float(row["x-coordinate"]), float(row["y-coordinate"])
                has_ball = row["visibility"] > 0 and 0 < x < W and 0 < y < H
                shutil.copy(os.path.join(clip, row["file name"]), f"{DST}/images/{name}.jpg")
                with open(f"{DST}/labels/{name}.txt", "w") as f:
                    if has_ball:
                        f.write(f"0 {x / W:.6f} {y / H:.6f} {BOX / W:.6f} {BOX / H:.6f}\n")
                n_ball += has_ball
                n_bg += not has_ball
                paths.append(os.path.abspath(f"{DST}/images/{name}.jpg"))
    return paths, n_ball, n_bg


def main():
    train_tn, tb, tg = convert(TRAIN_GAMES, TRAIN_STRIDE)
    val_tn, vb, vg = convert(VAL_GAMES, VAL_STRIDE)
    base_train = open(f"{DST}/train.txt").read().split()
    base_val = open(f"{DST}/val.txt").read().split()
    with open(f"{DST}/train_v4.txt", "w") as f:
        f.write("\n".join(base_train + train_tn) + "\n")
    with open(f"{DST}/val_v4.txt", "w") as f:
        f.write("\n".join(base_val + val_tn) + "\n")
    with open(f"{DST}/data_v4.yaml", "w") as f:
        f.write("path: data/ball_dataset\ntrain: train_v4.txt\nval: val_v4.txt\n\nnames:\n  0: tennis_ball\n")
    print(f"학습: 기존 {len(base_train)}장 + TrackNet {len(train_tn)}장(공 {tb}, 배경 {tg}) = {len(base_train) + len(train_tn)}장")
    print(f"검증: 기존 {len(base_val)}장 + TrackNet {len(val_tn)}장(공 {vb}, 배경 {vg}) = {len(base_val) + len(val_tn)}장")


if __name__ == "__main__":
    main()
