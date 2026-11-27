"""보고서용 그래프 생성과 원본 결과 보관.

- archive/    : 실험 원본 결과(학습 곡선·PR 곡선·혼동행렬·results.csv·args.yaml, 추적 라운드별 그래프)를 그대로 보관.
                학습 결과는 저장소 밖(Ultralytics runs 폴더, 환경변수 RUNS_ROOT 로 지정)에 생기므로 여기로 복사해 둔다.
- comparison/ : 보고서에 넣을 비교 그래프.
없는 입력은 건너뛰므로 실험이 끝날 때마다 다시 실행하면 된다.

사용: python src/make_report_figures.py
"""
import glob
import os
import re
import shutil

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# 학습 결과(Ultralytics runs) 폴더. 다른 위치에 있으면 환경변수 RUNS_ROOT 로 지정
RUNS_ROOT = os.environ.get("RUNS_ROOT", os.path.join("runs", "detect", "runs", "ball_detection"))
OUT = "docs/report"
BALL = f"{OUT}/01_ball_detection"
TRACK = f"{OUT}/02_ball_tracking"
PLAYER = f"{OUT}/03_player_model"
INT8 = f"{OUT}/04_int8"

# 보고서 이름 -> 학습 폴더 이름 (v1 은 같은 이름으로 여러 번 학습해 v1-4 가 최종)
RUNS = {"v1": "v1-4", "v2": "v2", "v3": "v3", "v3a": "v3a",
        "v4_yolo11n": "v4_yolo11n", "v4_yolo26n": "v4_yolo26n", "v4_yolo11s": "v4_yolo11s", "v4_yolo26s": "v4_yolo26s"}
KEEP = ["results.csv", "results.png", "args.yaml", "BoxPR_curve.png", "BoxF1_curve.png",
        "BoxP_curve.png", "BoxR_curve.png", "confusion_matrix.png", "confusion_matrix_normalized.png",
        "val_batch0_pred.jpg", "val_batch0_labels.jpg"]
COLORS = {"v1": "#8a96a8", "v2": "#1f5fd0", "v3": "#e9a23b", "v3a": "#c0392b",
          "v4_yolo11n": "#2a9d8f", "v4_yolo26n": "#7b4fd0", "v4_yolo11s": "#14756a", "v4_yolo26s": "#4a2c9a"}


def ensure(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def archive_runs():
    found = {}
    for name, folder in RUNS.items():
        src = os.path.join(RUNS_ROOT, folder)
        if not os.path.isdir(src):
            continue
        dst = f"{BALL}/archive/{name}"
        ensure(dst)
        for f in KEEP:
            if os.path.exists(os.path.join(src, f)):
                shutil.copy(os.path.join(src, f), dst)
        found[name] = os.path.join(dst, "results.csv")
        print(f"보관: {name} <- {src}")
    return found


def load_curves(found):
    out = {}
    for name, path in found.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.columns = [c.strip() for c in df.columns]
            out[name] = df
    return out


def fig_training_overlay(curves):
    if not curves:
        return
    ensure(f"{BALL}/comparison")
    panels = [("metrics/mAP50(B)", "검증 mAP50"), ("metrics/recall(B)", "검증 재현율"),
              ("train/box_loss", "학습 box 손실"), ("val/box_loss", "검증 box 손실")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (col, title) in zip(axes.ravel(), panels):
        for name, df in curves.items():
            if col in df:
                ax.plot(df["epoch"], df[col], label=name, color=COLORS.get(name), lw=1.6)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("공 검출 모델 학습 곡선 비교 (검증 데이터가 버전마다 달라 절대값 비교는 주의: v2 배경 추가, v4 TrackNet 추가)", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{BALL}/comparison/training_curves_overlay.png", dpi=160)
    plt.close()

    rows = []
    for name, df in curves.items():
        if "metrics/mAP50(B)" in df:
            i = int(df["metrics/mAP50(B)"].idxmax())
            rows.append({"모델": name, "학습 epoch 수": len(df), "최고 mAP50": round(df.loc[i, "metrics/mAP50(B)"], 3),
                         "최고 epoch": int(df.loc[i, "epoch"]), "검증 재현율(최고 epoch)": round(df.loc[i, "metrics/recall(B)"], 3),
                         "검증 정밀도(최고 epoch)": round(df.loc[i, "metrics/precision(B)"], 3)})
    pd.DataFrame(rows).to_csv(f"{BALL}/comparison/training_summary.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).to_string(index=False))


def fig_detection_v1_to_v3a():
    """results/v2_ball_experiment.md 에 기록된 video9 결과(conf 0.35, 20px 이내 적중 / 공 없는 프레임 오검출)."""
    ensure(f"{BALL}/comparison")
    hit, fp = [92.9, 95.3, 92.9, 99.4], [37.6, 15.6, 21.8, 31.6]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(x - 0.19, hit, 0.38, label="적중률 (높을수록 좋음)", color="#1f5fd0")
    ax.bar(x + 0.19, fp, 0.38, label="오검출률 (낮을수록 좋음)", color="#c0392b")
    for i in range(4):
        ax.text(x[i] - 0.19, hit[i] + 1, f"{hit[i]:.1f}", ha="center", fontsize=8)
        ax.text(x[i] + 0.19, fp[i] + 1, f"{fp[i]:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["v1\n배경 0장", "v2 (채택)\n+배경 571", "v3\n+공개 1,457", "v3a\n+배경 491"])
    ax.set_ylabel("%")
    ax.set_title("공 검출 v1~v3a 비교 (video9, conf 0.35)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(0, 110)
    plt.tight_layout()
    plt.savefig(f"{BALL}/comparison/v1_to_v3a_video9.png", dpi=160)
    plt.close()


def fig_tracking():
    ensure(f"{TRACK}/archive", f"{TRACK}/comparison")
    for f in glob.glob("docs/ball_tracking/round*.png") + glob.glob("results/ball_tracking_round*_log.txt"):
        shutil.copy(f, f"{TRACK}/archive")
    log = "results/ball_tracking_round3_log.txt"
    if not os.path.exists(log):
        return
    rows = []
    for line in open(log, encoding="utf-8"):
        m = re.match(r"^(\S+(?: \S+)? \((?:조정|시험)\))\s+(.+?)\s{2,}([\d.]+)%\s+([\d.]+)%\s+([\d.]+)\s", line)
        if m:
            rows.append((m.group(1), m.group(2).strip(), float(m.group(3)), float(m.group(4)), float(m.group(5))))
    if not rows:
        return
    df = pd.DataFrame(rows, columns=["group", "system", "recall", "precision", "f1"])
    df["system"] = df["system"].replace({"추적(조정된 파라미터)": "추적 3차(최종 선택)"})
    df.to_csv(f"{TRACK}/comparison/tracking_results.csv", index=False, encoding="utf-8-sig")
    groups = list(dict.fromkeys(df["group"]))
    systems = list(dict.fromkeys(df["system"]))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, col, title in zip(axes, ["recall", "precision", "f1"], ["재현율 (%)", "정밀도 (%)", "F1"]):
        w = 0.8 / len(systems)
        for k, s in enumerate(systems):
            vals = [df[(df.group == g) & (df.system == s)][col].iloc[0] for g in groups]
            ax.bar(np.arange(len(groups)) + (k - len(systems) / 2 + 0.5) * w, vals, w, label=s)
        ax.set_xticks(np.arange(len(groups)))
        ax.set_xticklabels([g.replace(" (", "\n(") for g in groups], fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("공 추적: 프레임별 최고 검출 대 추적(1·2·3차 설정) — 조정용/시험용 분리", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{TRACK}/comparison/tracking_rounds.png", dpi=160)
    plt.close()


def fig_int8():
    ensure(f"{INT8}/comparison", f"{INT8}/archive")
    names = ["PyTorch FP32", "OpenVINO FP32", "OpenVINO INT8"]
    ms, mb = [72.2, 85.7, 50.0], [5.6, 10.9, 3.5]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, vals, title, unit in zip(axes, [ms, mb], ["프레임당 추론 시간", "모델 크기"], ["ms (낮을수록 좋음)", "MB (낮을수록 좋음)"]):
        bars = ax.bar(names, vals, color=["#8a96a8", "#e9a23b", "#1f5fd0"])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.02, f"{v}", ha="center", fontsize=9)
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("INT8 경량화 (CPU Xeon W-2223, 입력 544x960, 300프레임) — 출처: results/int8_experiment.md", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{INT8}/comparison/int8_speed_size.png", dpi=160)
    plt.close()
    for f in ["results/int8_experiment.md", "results/int8_run1_square960_log.txt", "results/int8_service_compare_log.txt"]:
        if os.path.exists(f):
            shutil.copy(f, f"{INT8}/archive")


def collect_player_comparison():
    if glob.glob("docs/model_comparison/*.png"):
        ensure(f"{PLAYER}/archive", f"{PLAYER}/comparison")
        for f in glob.glob("docs/model_comparison/*.png"):
            shutil.copy(f, f"{PLAYER}/comparison")
        if os.path.exists("results/player_model_comparison_log.txt"):
            shutil.copy("results/player_model_comparison_log.txt", f"{PLAYER}/archive")


def main():
    ensure(BALL, TRACK, INT8)
    found = archive_runs()
    fig_training_overlay(load_curves(found))
    fig_detection_v1_to_v3a()
    fig_tracking()
    fig_int8()
    collect_player_comparison()
    print("완료:", OUT)


if __name__ == "__main__":
    main()
