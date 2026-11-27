"""공 검출 모델 비교(시험): 저장해 둔 검출 후보(dets_*.pkl)를 정답과 비교해 v2, v4(11n), v4(26n)를 평가한다.

공정한 기준
- TrackNet game6~10: 세 모델 모두 학습·검증에 쓰지 않은 데이터(처음 보는 방송 영상). 가장 중요한 시험.
- video9, video10: v2 와 v4 모두 best epoch 선택(검증)에 쓰인 영상. 참고용.
- TrackNet game1~4: v4 의 학습 데이터(v2 에는 처음 보는 데이터). 비교에는 쓰지 않고 참고로만 표시.
지표: 정답 20px 이내(1920 너비 기준, 해상도에 비례)를 적중으로 본다.
- 재현율 = 적중 / 공이 보이는 프레임, 정밀도 = 적중 / 출력 프레임, 오출력률 = 공 없는 프레임 중 출력 비율
- 프레임별 최고 점수 검출(임계값 sweep) 과 공 추적(공 추적 3차 최종 설정) 두 가지로 평가.
사용: python src/compare_ball_models.py   (먼저 src/ball_track_detect.py 로 각 모델의 dets_*.pkl 생성)
"""
import os
import pickle
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "webapp", "backend"))
from ball_tracker import DETECTED, TrackerConfig, track_ball  # noqa: E402

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

CACHES = {"v2 (기존)": "data/ball_tracking/dets_v2.pkl", "v4 · yolo11n": "data/ball_tracking/dets_v4_yolo11n.pkl",
          "v4 · yolo26n": "data/ball_tracking/dets_v4_yolo26n.pkl"}
COLORS = {"v2 (기존)": "#1f5fd0", "v4 · yolo11n": "#2a9d8f", "v4 · yolo26n": "#7b4fd0"}
TRACKER = TrackerConfig(max_gap=3, start_cost=5.0, dist_weight=1.0, min_peak_conf=0.3, conf_bias=1.2)
THR_1920, TOPK = 20, 8
THRESHOLDS = np.round(np.arange(0.1, 0.91, 0.05), 2)
OUT = "docs/report/01_ball_detection/comparison"


def load(path):
    data = pickle.load(open(path, "rb"))["data"]
    for seq in data.values():
        seq["dets"] = [d[np.argsort(-d[:, 2])[:TOPK]] if len(d) else d for d in seq["dets"]]
    return data


def game(key):
    return int(key.split("/")[0][4:])


def groups_of(data):
    tn = [k for k in data if k.startswith("game")]
    return {"video9 (참고)": ["video9"], "video10 (참고)": ["video10"],
            "TrackNet game6-10 (시험)": [k for k in tn if game(k) >= 6],
            "TrackNet game1-4 (v4 학습에 사용·참고)": [k for k in tn if game(k) <= 4]}


def baseline_output(dets, thr):
    out = np.zeros((len(dets), 4), np.float32)
    for t, d in enumerate(dets):
        d = d[d[:, 2] >= thr] if len(d) else d
        if len(d):
            k = int(np.argmax(d[:, 2]))
            out[t] = (d[k, 0], d[k, 1], d[k, 2], DETECTED)
    return out


def score(out, seq):
    gt = seq["gt"]
    thr = THR_1920 * seq["W"] / 1920
    vis, has = gt[:, 0] > 0, out[:, 3] > 0
    err = np.hypot(out[:, 0] - gt[:, 1], out[:, 1] - gt[:, 2])
    return {"n_gt": int(vis.sum()), "n_out": int(has.sum()), "hit": int((has & vis & (err <= thr)).sum()),
            "fp": int((has & ~vis).sum()), "n_nov": int((~vis).sum())}


def metrics(parts):
    a = {k: sum(p[k] for p in parts) for k in parts[0]}
    p, r = a["hit"] / max(a["n_out"], 1), a["hit"] / max(a["n_gt"], 1)
    return {"recall": r, "precision": p, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "fp_rate": a["fp"] / a["n_nov"] if a["n_nov"] else float("nan")}


def evaluate(data, keys, fn):
    return metrics([score(fn(data[k]["dets"], data[k]), data[k]) for k in keys])


def main():
    all_data = {name: load(path) for name, path in CACHES.items() if os.path.exists(path)}
    groups = groups_of(next(iter(all_data.values())))
    log, rows, curves = [], [], {}

    def say(s=""):
        print(s, flush=True)
        log.append(s)

    say(f"적중 기준 {THR_1920}px(1920 너비 기준), 후보 프레임당 상위 {TOPK}개")
    say(f"공 추적 설정(3차 최종): {TRACKER}")
    for g, keys in groups.items():
        say(f"  {g}: 시퀀스 {len(keys)}개")
    say(f"\n{'그룹':<34}{'모델':<14}{'방식':<20}{'재현율':>8}{'정밀도':>8}{'F1':>7}{'오출력률':>9}")
    for g, keys in groups.items():
        for name, data in all_data.items():
            curves[(g, name)] = [evaluate(data, keys, lambda d, s, t=t: baseline_output(d, t)) for t in THRESHOLDS]
            systems = {"프레임별 검출 conf 0.35": curves[(g, name)][int(np.where(THRESHOLDS == 0.35)[0][0])],
                       "프레임별 검출 conf 0.25": curves[(g, name)][int(np.where(THRESHOLDS == 0.25)[0][0])],
                       "공 추적": evaluate(data, keys, lambda d, s: track_ball(d, s["W"], TRACKER))}
            for sysname, m in systems.items():
                rows.append({"그룹": g, "모델": name, "방식": sysname, **m})
                say(f"{g:<34}{name:<14}{sysname:<20}{m['recall'] * 100:>7.1f}%{m['precision'] * 100:>7.1f}%"
                    f"{m['f1']:>7.3f}{m['fp_rate'] * 100:>8.1f}%")
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/test_comparison.csv", index=False, encoding="utf-8-sig")
    with open("results/ball_model_comparison_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log))

    shown = [g for g in groups if "학습에 사용" not in g]
    fig, axes = plt.subplots(1, len(shown), figsize=(5.2 * len(shown), 4.6))
    for ax, g in zip(axes, shown):
        for name in all_data:
            c = curves[(g, name)]
            ax.plot([m["recall"] for m in c], [m["precision"] for m in c], "o-", ms=3, label=name, color=COLORS[name])
        ax.set_title(g)
        ax.set_xlabel("재현율 (공을 찾은 비율)")
        ax.set_ylabel("정밀도 (표시한 것 중 맞은 비율)")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("공 검출 모델 시험: 정밀도–재현율 곡선 (오른쪽 위일수록 좋음)", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{OUT}/test_pr_curves.png", dpi=160)
    plt.close()

    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, len(shown), figsize=(5.2 * len(shown), 4.4), sharey=True)
    for ax, g in zip(axes, shown):
        sub = df[(df["그룹"] == g) & (df["방식"] == "프레임별 검출 conf 0.25")]
        sub_t = df[(df["그룹"] == g) & (df["방식"] == "공 추적")]
        x = np.arange(len(all_data))
        ax.bar(x - 0.2, [sub[sub["모델"] == n]["recall"].iloc[0] * 100 for n in all_data], 0.4, label="프레임별 검출(conf 0.25)",
               color="#8a96a8")
        ax.bar(x + 0.2, [sub_t[sub_t["모델"] == n]["recall"].iloc[0] * 100 for n in all_data], 0.4, label="공 추적", color="#1f5fd0")
        ax.set_xticks(x)
        ax.set_xticklabels(list(all_data), fontsize=8)
        ax.set_title(g, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("재현율 (%)")
    axes[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{OUT}/test_recall_bars.png", dpi=160)
    plt.close()
    print("저장:", OUT)


def fair_groups(data):
    tn = [k for k in data if k.startswith("game")]
    tune = {"video9 (조정)": ["video9"], "TrackNet game5 (조정)": [k for k in tn if game(k) == 5]}
    test = {"TrackNet game6-10 (시험)": [k for k in tn if game(k) >= 6], "video10 (시험·참고)": ["video10"]}
    return tune, test


def main_fair():
    """모델마다 확신도 기준·추적 설정을 조정용 데이터(video9 + TrackNet game5)에서만 고른 뒤, 시험 데이터에서 처음 평가한다."""
    import itertools

    all_data = {name: load(path) for name, path in CACHES.items() if os.path.exists(path)}
    tune, test = fair_groups(next(iter(all_data.values())))
    grid = [dict(conf_bias=a, min_peak_conf=b, trim_conf=c, start_cost=5.0, max_gap=3, dist_weight=1.0)
            for a, b, c in itertools.product([0.6, 1.2, 2.2], [0.15, 0.30, 0.45], [0.0, 0.30])]
    log, rows = [], []

    def say(s=""):
        print(s, flush=True)
        log.append(s)

    say("공평한 비교: 모델마다 설정을 조정용(video9 + TrackNet game5)에서만 고르고, 시험(TrackNet game6-10, video10)에서 평가")
    say(f"추적 설정 후보 {len(grid)}개 (conf_bias, min_peak_conf, trim_conf 조합), 확신도 기준 후보 {len(THRESHOLDS)}개")
    objective = lambda fn, data: float(np.mean([evaluate(data, keys, fn)["f1"] for keys in tune.values()]))
    chosen = {}
    for name, data in all_data.items():
        thr = max(THRESHOLDS, key=lambda t: objective(lambda d, s, t=t: baseline_output(d, t), data))
        best_cfg, best = None, -1.0
        for p in grid:
            cfg = TrackerConfig(**p)
            obj = objective(lambda d, s, c=cfg: track_ball(d, s["W"], c), data)
            if obj > best:
                best_cfg, best = p, obj
        chosen[name] = (float(thr), best_cfg)
        say(f"\n[{name}] 선택: 프레임별 검출 conf {thr:.2f} / 추적 {best_cfg} (조정용 평균 F1 {best:.3f})")

    say(f"\n{'그룹':<26}{'모델':<14}{'방식':<22}{'재현율':>8}{'정밀도':>8}{'F1':>7}{'오출력률':>9}")
    for g, keys in {**tune, **test}.items():
        for name, data in all_data.items():
            thr, p = chosen[name]
            systems = {f"프레임별 검출 conf {thr:.2f}": evaluate(data, keys, lambda d, s: baseline_output(d, thr)),
                       "공 추적(모델별 조정)": evaluate(data, keys, lambda d, s: track_ball(d, s["W"], TrackerConfig(**p)))}
            for sysname, m in systems.items():
                rows.append({"그룹": g, "모델": name, "방식": sysname, **m})
                say(f"{g:<26}{name:<14}{sysname:<22}{m['recall'] * 100:>7.1f}%{m['precision'] * 100:>7.1f}%"
                    f"{m['f1']:>7.3f}{m['fp_rate'] * 100:>8.1f}%")
    os.makedirs(OUT, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/fair_comparison.csv", index=False, encoding="utf-8-sig")
    with open("results/ball_model_comparison_fair_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, g in zip(axes, test):
        x = np.arange(len(all_data))
        for k, (kind, color) in enumerate([("프레임별 검출", "#8a96a8"), ("공 추적", "#1f5fd0")]):
            vals = [df[(df["그룹"] == g) & (df["모델"] == n) & (df["방식"].str.startswith(kind))]["f1"].iloc[0] for n in all_data]
            bars = ax.bar(x + (k - 0.5) * 0.38, vals, 0.38, label=kind + "(모델별 조정)", color=color)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(list(all_data))
        ax.set_title(g)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("종합 점수 F1 (높을수록 좋음)")
    axes[0].legend(fontsize=8)
    fig.suptitle("공평한 비교: 모델마다 설정을 조정용 데이터에서만 고른 뒤 시험 데이터에서 평가", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fair_f1.png", dpi=160)
    plt.close()
    print("저장:", OUT)


if __name__ == "__main__":
    main_fair() if len(sys.argv) > 1 and sys.argv[1] == "fair" else main()
