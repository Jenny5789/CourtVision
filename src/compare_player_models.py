"""선수 검출 모델 비교: 같은 영상·같은 설정으로 사전학습 모델(COCO, person)만 비교한다. 학습은 하지 않는다.

정확도
- 정답: data/raw/archive/{video}_player.csv 의 선수 위치(player_l, player_r). 화면 밖 좌표(x>1900, y<20)는 "선수 없음".
- 적중: 정답 점이 검출 박스(가로·세로 10% 여유 포함) 안에 들어오면 적중. 양쪽 선수 각각 센다.
- 추가 검출: 정답 점이 하나도 들어 있지 않은 person 박스 수(프레임당). 관중·심판·정답에 없는 사람 포함이라 정밀도가 아니라 참고 지표.
속도
- GPU: 영상 전체에서 프레임당 추론 시간(앞 10프레임 제외), 배치 1.
- CPU: 영상 앞부분 40프레임(앞 5프레임 제외).
조건: conf 0.3, imgsz 960 (서비스와 동일), classes=[0].

주의: 정답 점이 선수의 어느 부위인지는 데이터에 명시되어 있지 않다. 박스 안 포함 여부만 보므로 모델 간 상대 비교용이다.
영상 2개, 측정 1회. video9·10 은 공 검출 모델 선택에 쓰인 영상이지만 선수 검출은 사전학습 모델이라 학습 누수는 없다.
"""
import gc
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

MODELS = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolov8n.pt", "yolo26n.pt", "yolo26s.pt"]
VIDEOS = ["video9", "video10"]
CONF, IMGSZ, MARGIN = 0.3, 960, 0.10
CPU_FRAMES = 40
OUT_DIR = "docs/model_comparison"


def gt_points(row):
    pts = []
    for x, y in ((row.player_l_x, row.player_l_y), (row.player_r_x, row.player_r_y)):
        if not (x > 1900 and y < 20):
            pts.append((x, y))
    return pts


def inside(box, pt):
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * MARGIN, (y2 - y1) * MARGIN
    return x1 - mx <= pt[0] <= x2 + mx and y1 - my <= pt[1] <= y2 + my


def evaluate(model, video):
    gt = pd.read_csv(f"data/raw/archive/{video}_player.csv")
    cap = cv2.VideoCapture(f"data/raw/archive/{video}.mp4")
    hits = n_gt = extras = frames = 0
    times = []
    for i, row in enumerate(gt.itertuples()):
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        res = model.predict(frame, classes=[0], conf=CONF, imgsz=IMGSZ, verbose=False)[0]
        dt = (time.perf_counter() - t0) * 1000
        if i >= 10:
            times.append(dt)
        boxes = res.boxes.xyxy.cpu().numpy()
        points = gt_points(row)
        for p in points:
            n_gt += 1
            hits += any(inside(b, p) for b in boxes)
        extras += sum(not any(inside(b, p) for p in points) for b in boxes)
        frames += 1
    cap.release()
    return {"hit": hits / max(n_gt, 1) * 100, "n_gt": n_gt, "extra": extras / max(frames, 1), "gpu_ms": float(np.mean(times))}


def cpu_speed(path):
    model = YOLO(path)
    cap = cv2.VideoCapture("data/raw/archive/video9.mp4")
    times = []
    for i in range(CPU_FRAMES):
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        model.predict(frame, classes=[0], conf=CONF, imgsz=IMGSZ, device="cpu", verbose=False)
        if i >= 5:
            times.append((time.perf_counter() - t0) * 1000)
    cap.release()
    return float(np.mean(times))


def main():
    rows = []
    for path in MODELS:
        model = YOLO(path)
        row = {"model": path.replace(".pt", ""), "size_mb": os.path.getsize(str(model.ckpt_path)) / 1e6}
        gpu = []
        for video in VIDEOS:
            r = evaluate(model, video)
            row[f"hit_{video}"], row[f"extra_{video}"] = r["hit"], r["extra"]
            gpu.append(r["gpu_ms"])
            print(f"{path} {video}: 적중 {r['hit']:.1f}% ({r['n_gt']}건), 추가검출 {r['extra']:.2f}/프레임, GPU {r['gpu_ms']:.1f}ms", flush=True)
        row["gpu_ms"] = float(np.mean(gpu))
        del model
        gc.collect()
        torch.cuda.empty_cache()
        row["cpu_ms"] = cpu_speed(path)
        print(f"{path} CPU {row['cpu_ms']:.0f}ms", flush=True)
        rows.append(row)
    df = pd.DataFrame(rows)
    print("\n" + df.round(2).to_string(index=False))
    with open("results/player_model_comparison_log.txt", "w", encoding="utf-8") as f:
        f.write(f"conf {CONF}, imgsz {IMGSZ}, margin {MARGIN}, CPU 프레임 {CPU_FRAMES}\n")
        f.write(df.round(2).to_string(index=False))
    plot(df)


def plot(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    os.makedirs(OUT_DIR, exist_ok=True)
    df["hit_avg"] = df[[f"hit_{v}" for v in VIDEOS]].mean(axis=1)
    colors = ["#1f5fd0", "#2a9d8f", "#e9a23b", "#c0392b", "#7b4fd0", "#4a2c9a"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, col, label in zip(axes, ["cpu_ms", "gpu_ms"], ["CPU 프레임당 시간 (ms)", "GPU 프레임당 시간 (ms)"]):
        for (_, r), c in zip(df.iterrows(), colors):
            ax.scatter(r[col], r["hit_avg"], s=r["size_mb"] * 14, color=c, alpha=0.8, label=r["model"])
            ax.annotate(r["model"], (r[col], r["hit_avg"]), textcoords="offset points", xytext=(8, 6), fontsize=9)
        ax.set_xlabel(label)
        ax.set_ylabel("선수 적중률 (%, 영상 평균)")
        ax.grid(alpha=0.3)
    axes[0].set_title("정확도 대 속도 (점 크기 = 모델 용량)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/accuracy_vs_speed.png", dpi=160)
    plt.close()

    x = np.arange(len(df))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, v in enumerate(VIDEOS):
        ax.bar(x + (i - 0.5) * w, df[f"hit_{v}"], w, label=v)
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"])
    ax.set_ylabel("선수 적중률 (%)")
    ax.set_ylim(max(0, df[[f"hit_{v}" for v in VIDEOS]].min().min() - 10), 100.5)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/hit_rate.png", dpi=160)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, v in enumerate(VIDEOS):
        ax.bar(x + (i - 0.5) * w, df[f"extra_{v}"], w, label=v)
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"])
    ax.set_ylabel("정답 외 추가 검출 (건/프레임)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/extra_detections.png", dpi=160)
    plt.close()
    print("그래프 저장:", OUT_DIR)


if __name__ == "__main__":
    sys.exit(main())
