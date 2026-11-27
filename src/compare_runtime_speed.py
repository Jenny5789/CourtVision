"""실행 환경별 CPU 속도 비교: 같은 모델(yolo11n, yolo26n)을 PyTorch / ONNX Runtime / OpenVINO 로 재서 비교한다.

배경: Ultralytics 문서는 YOLO26n 이 ONNX Runtime CPU 에서 YOLO11n 보다 빠르다고 하지만(제조사 측정),
      우리가 PyTorch CPU 로 잰 값은 반대였다(11n 85ms, 26n 289ms). 환경 차이인지 확인한다.
조건: 사전학습 모델(COCO, person), FP32, 입력 544x960(16:9 영상에서 imgsz 960 일 때의 letterbox 크기와 같음, INT8 실험과 동일),
      video9 의 같은 프레임 40장(앞 5장은 워밍업), CPU 만 사용(GPU 미사용), 배치 1.
측정 신뢰도를 위해
- 측정 직전에 PC CPU 사용률이 MAX_IDLE_LOAD 아래로 내려올 때까지 기다린다(최대 WAIT_SEC).
- 같은 측정을 REPEAT 번 반복하고 각 회차의 중앙값 중 가장 작은 값(방해가 가장 적었던 값)을 대표값으로 쓴다. 모든 회차 값도 기록한다.
측정 중 PC 가 바빴는지 알 수 있도록 회차마다 측정 직전·직후 CPU 사용률을 기록한다.
사용: python src/compare_runtime_speed.py
"""
import os
import shutil
import time

import cv2
import numpy as np
import pandas as pd
import psutil
from ultralytics import YOLO

MODELS = ["yolo11n", "yolo26n"]
FRAMES, WARMUP, REPEAT = 40, 5, 3
IMGSZ = [544, 960]
MAX_IDLE_LOAD, WAIT_SEC = 20.0, 180
WORK = "data/runtime_bench"
OUT = "docs/report/03_player_model/comparison"


def load_frames():
    cap = cv2.VideoCapture("data/raw/archive/video9.mp4")
    frames = []
    for _ in range(FRAMES):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def wait_idle():
    t0 = time.time()
    while True:
        load = psutil.cpu_percent(interval=3)
        if load < MAX_IDLE_LOAD:
            return load
        if time.time() - t0 > WAIT_SEC:
            print(f"  경고: {WAIT_SEC}초를 기다려도 CPU 사용률이 {load:.0f}% 라 그대로 측정합니다", flush=True)
            return load
        print(f"  PC 가 바쁨(CPU {load:.0f}%) - 기다리는 중", flush=True)


def bench_once(model, frames):
    times = []
    for i, f in enumerate(frames):
        t0 = time.perf_counter()
        model.predict(f, classes=[0], conf=0.3, imgsz=IMGSZ, device="cpu", verbose=False)
        if i >= WARMUP:
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times)), float(np.std(times))


def bench(model, frames):
    runs = []
    for _ in range(REPEAT):
        before = wait_idle()
        med, std = bench_once(model, frames)
        after = psutil.cpu_percent(interval=0.5)
        runs.append({"median": med, "std": std, "load_before": before, "load_after": after})
    return runs


def main():
    os.makedirs(WORK, exist_ok=True)
    frames = load_frames()
    rows = []
    for name in MODELS:
        src = str(YOLO(f"{name}.pt").ckpt_path)
        local = os.path.join(WORK, f"{name}.pt")
        shutil.copy(src, local)
        base = YOLO(local)
        exports = {"PyTorch": local}
        for fmt, label in [("onnx", "ONNX Runtime"), ("openvino", "OpenVINO")]:
            try:
                exports[label] = str(base.export(format=fmt, imgsz=IMGSZ, dynamic=False, verbose=False)).rstrip("\\/")
            except Exception as e:
                print(f"[{name}] {label} 변환 실패: {e}", flush=True)
        for label, path in exports.items():
            m = YOLO(path, task="detect") if label != "PyTorch" else YOLO(path)
            runs = bench(m, frames)
            best = min(r["median"] for r in runs)
            rows.append({"모델": name, "환경": label, "ms": round(best, 1),
                         "3회 중앙값": " / ".join(f"{r['median']:.0f}" for r in runs),
                         "측정 직전 CPU%": " / ".join(f"{r['load_before']:.0f}" for r in runs),
                         "측정 직후 CPU%": " / ".join(f"{r['load_after']:.0f}" for r in runs)})
            print(f"{name} {label}: 대표 {best:.1f} ms | 3회 {rows[-1]['3회 중앙값']} | CPU 직전 {rows[-1]['측정 직전 CPU%']} 직후 {rows[-1]['측정 직후 CPU%']}", flush=True)
    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    with open("results/runtime_comparison_log.txt", "w", encoding="utf-8") as f:
        f.write(f"CPU, FP32, 입력 {IMGSZ}, 프레임 {len(frames)}장(워밍업 {WARMUP}), {REPEAT}회 반복 중 최소 중앙값\n" + df.to_string(index=False))
    plot(df)


def plot(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    os.makedirs(OUT, exist_ok=True)
    runtimes = ["PyTorch", "ONNX Runtime", "OpenVINO"]
    x = np.arange(len(runtimes))
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for k, (name, color) in enumerate(zip(MODELS, ["#1f5fd0", "#7b4fd0"])):
        vals = []
        for r in runtimes:
            sub = df[(df["모델"] == name) & (df["환경"] == r)]
            vals.append(sub["ms"].iloc[0] if len(sub) else 0)
        bars = ax.bar(x + (k - 0.5) * 0.38, vals, 0.38, label=name, color=color)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 3, f"{v:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(runtimes)
    ax.set_ylabel("프레임당 CPU 추론 시간 (ms, 낮을수록 좋음)")
    ax.set_title("실행 환경별 CPU 속도: YOLO11n 대 YOLO26n (FP32, 입력 544x960, 3회 중 최소)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/runtime_speed.png", dpi=160)
    plt.close()
    print("그래프 저장:", OUT)


if __name__ == "__main__":
    main()
