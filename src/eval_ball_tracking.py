"""공 추적 평가: 프레임별 최고 검출만 쓰는 방식(기준) vs 추적(webapp/backend/ball_tracker.py).

전제: python src/ball_track_detect.py 로 data/ball_tracking/dets_v2.pkl 을 먼저 만든다(학습 없음, v2 검출 후보).

데이터 분리 (파라미터를 평가 데이터에 맞춰 고르지 않기 위해)
- 조정용: video9, TrackNet game1~5  -> 추적 파라미터와 기준 방식의 conf 임계값을 여기서만 고른다.
- 시험용: video10, TrackNet game6~10 -> 고른 값으로 처음 평가한다.
주의: video9·10 은 v2 의 best epoch 선택에 쓰였고, TrackNet 은 v2 학습에 쓰지 않은 방송 영상(도메인이 다름).

지표 (정답 20px 이내(1920 너비 기준, 해상도에 비례) 를 적중으로 본다)
- 재현율 = 적중 / 공이 보이는 프레임, 정밀도 = 적중 / 출력한 프레임(공이 없는 프레임의 출력은 오출력)
- 오출력률 = 공이 없는 프레임 중 출력한 비율(video 계열만 의미 있음. TrackNet 은 거의 모든 프레임에 공이 있음)
- 보간 정확도 = 보간으로 채운 프레임 중 적중 비율
"""
import itertools
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "webapp", "backend"))
from ball_tracker import DETECTED, INTERPOLATED, TrackerConfig, track_ball  # noqa: E402

CACHE = "data/ball_tracking/dets_v2.pkl"
OUT_DIR = "docs/ball_tracking"
THR_1920 = 20
TOPK = 8
# 라운드 1: 속도·비용 파라미터만 탐색(1차 결과에서 아마추어 영상의 정밀도가 크게 떨어짐)
# 라운드 2: 조정용 video9 에서 이미 보인 정밀도 하락을 줄이기 위해 구간 조건(min_peak_conf, min_nodes, max_gap, start_cost)을 탐색
ROUND = int(sys.argv[1]) if len(sys.argv) > 1 else 1
TUNE_STRIDE = 1 if ROUND == 1 else 2     # 라운드 2는 조정용 TrackNet 클립을 격 클립으로 줄여 탐색 시간을 줄임
# 라운드 3: 라운드 2 진단(조정용 video9)에서 오출력이 "공이 사라진 뒤 낮은 점수 오검출로 구간이 늘어나는 것"으로 확인되어
#           구간 끝을 확실한 검출로 제한(trim_conf)하고 낮은 점수 후보의 보상(conf_bias)을 낮춰 탐색
ROUND1_CONFIG = dict(start_cost=5.0, max_speed=0.12, dist_weight=1.0, still_cost=1.0)
ROUND2_CONFIG = dict(start_cost=5.0, max_speed=0.12, dist_weight=1.0, still_cost=1.0, min_nodes=3, max_gap=3,
                     min_peak_conf=0.3)


def load():
    data = pickle.load(open(CACHE, "rb"))["data"]
    for seq in data.values():
        seq["dets"] = [d[np.argsort(-d[:, 2])[:TOPK]] if len(d) else d for d in seq["dets"]]
    game = lambda k: int(k.split("/")[0][4:])
    groups = {
        "video9 (조정)": ["video9"],
        "TrackNet game1-5 (조정)": [k for k in data if k.startswith("game") and game(k) <= 5],
        "video10 (시험)": ["video10"],
        "TrackNet game6-10 (시험)": [k for k in data if k.startswith("game") and game(k) > 5],
    }
    return data, groups


def baseline_output(dets, conf_thr):
    out = np.zeros((len(dets), 4), np.float32)
    for t, d in enumerate(dets):
        d = d[d[:, 2] >= conf_thr] if len(d) else d
        if len(d):
            k = int(np.argmax(d[:, 2]))
            out[t] = (d[k, 0], d[k, 1], d[k, 2], DETECTED)
    return out


def score(out, seq):
    gt = seq["gt"]
    thr = THR_1920 * seq["W"] / 1920
    vis = gt[:, 0] > 0
    has = out[:, 3] > 0
    err = np.where(has, np.hypot(out[:, 0] - gt[:, 1], out[:, 1] - gt[:, 2]), np.inf)
    correct = has & vis & (err <= thr)
    interp = out[:, 3] == INTERPOLATED
    return {"n_gt": int(vis.sum()), "n_out": int(has.sum()), "hit": int(correct.sum()),
            "fp": int((has & ~vis).sum()), "n_nov": int((~vis).sum()),
            "n_int": int(interp.sum()), "hit_int": int((correct & interp).sum()), "err": err[vis]}


def aggregate(parts):
    agg = {k: sum(p[k] for p in parts) for k in ["n_gt", "n_out", "hit", "fp", "n_nov", "n_int", "hit_int"]}
    agg["err"] = np.concatenate([p["err"] for p in parts])
    p, r = agg["hit"] / max(agg["n_out"], 1), agg["hit"] / max(agg["n_gt"], 1)
    agg.update(precision=p, recall=r, f1=2 * p * r / (p + r) if p + r else 0.0,
               fp_rate=agg["fp"] / agg["n_nov"] if agg["n_nov"] else float("nan"),
               int_acc=agg["hit_int"] / agg["n_int"] if agg["n_int"] else float("nan"))
    return agg


def evaluate(data, keys, make_output):
    return aggregate([score(make_output(data[k]["dets"], data[k]), data[k]) for k in keys])


def main():
    data, groups = load()
    tune = ["video9 (조정)", "TrackNet game1-5 (조정)"]
    log = []

    def say(s=""):
        print(s, flush=True)
        log.append(s)

    say(f"후보: 프레임당 상위 {TOPK}개, 적중 기준 {THR_1920}px(1920 너비 기준)")
    for g, keys in groups.items():
        say(f"  {g}: 시퀀스 {len(keys)}개, 프레임 {sum(len(data[k]['dets']) for k in keys)}, "
            f"공 보이는 프레임 {int(sum((data[k]['gt'][:, 0] > 0).sum() for k in keys))}")

    objective = lambda res: float(np.mean([res[g]["f1"] for g in tune]))

    say("\n[기준 방식] 조정용에서 conf 임계값 선택")
    best_conf, best_obj = None, -1
    for conf in np.arange(0.1, 0.91, 0.05):
        res = {g: evaluate(data, groups[g], lambda d, s, c=conf: baseline_output(d, c)) for g in tune}
        obj = objective(res)
        if obj > best_obj:
            best_conf, best_obj = float(conf), obj
    say(f"  선택: conf {best_conf:.2f} (조정용 평균 F1 {best_obj:.3f})")

    say("\n[추적] 조정용에서 파라미터 격자 탐색")
    if ROUND == 1:
        grid = [dict(start_cost=a, max_speed=b, dist_weight=c, still_cost=d)
                for a, b, c, d in itertools.product([3.0, 5.0], [0.06, 0.12], [1.0, 3.0], [0.0, 1.0])]
    elif ROUND == 2:
        grid = [dict(start_cost=a, min_nodes=b, max_gap=c, min_peak_conf=d, max_speed=0.12, dist_weight=1.0)
                for a, b, c, d in itertools.product([5.0, 8.0], [3, 6], [3, 8], [0.3, 0.6])]
    else:
        grid = [dict(conf_bias=a, trim_conf=b, start_cost=5.0, min_nodes=3, max_gap=3, min_peak_conf=0.3,
                     max_speed=0.12, dist_weight=1.0)
                for a, b in itertools.product([0.5, 1.2, 2.2], [0.0, 0.35, 0.42, 0.46])]
    tune_keys = {g: (groups[g][::TUNE_STRIDE] if g.startswith("TrackNet") else groups[g]) for g in tune}
    best_cfg, best_obj = None, -1
    for params in grid:
        cfg = TrackerConfig(**params)
        res = {g: evaluate(data, tune_keys[g], lambda d, s, c=cfg: track_ball(d, s["W"], c)) for g in tune}
        obj = objective(res)
        say(f"  {params}: F1 {obj:.3f}")
        if obj > best_obj:
            best_cfg, best_obj = cfg, obj
    say(f"  선택: {best_cfg} (조정용 평균 F1 {best_obj:.3f})")

    systems = {
        "기준 conf 0.35": lambda d, s: baseline_output(d, 0.35),
        f"기준 conf {best_conf:.2f}(조정)": lambda d, s: baseline_output(d, best_conf),
        "추적(조정된 파라미터)": lambda d, s: track_ball(d, s["W"], best_cfg),
    }
    if ROUND >= 2:
        round1 = TrackerConfig(**ROUND1_CONFIG)
        systems["추적 1차 설정"] = lambda d, s: track_ball(d, s["W"], round1)
    if ROUND >= 3:
        round2 = TrackerConfig(**ROUND2_CONFIG)
        systems["추적 2차 설정"] = lambda d, s: track_ball(d, s["W"], round2)
    results = {}
    say("\n[최종 비교]")
    say(f"{'그룹':<28}{'방식':<22}{'재현율':>8}{'정밀도':>8}{'F1':>8}{'오출력률':>9}{'보간 정확도':>11}")
    for g, keys in groups.items():
        for name, fn in systems.items():
            r = results[(g, name)] = evaluate(data, keys, fn)
            say(f"{g:<28}{name:<22}{r['recall'] * 100:>7.1f}%{r['precision'] * 100:>7.1f}%{r['f1']:>8.3f}"
                f"{r['fp_rate'] * 100:>8.1f}%{r['int_acc'] * 100:>10.1f}%")
    os.makedirs("results", exist_ok=True)
    with open(f"results/ball_tracking_round{ROUND}_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log))
    plots(data, groups, best_conf, best_cfg, results, systems)


def plots(data, groups, best_conf, best_cfg, results, systems):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (g, keys) in zip(axes.ravel(), groups.items()):
        pr = [(r["recall"], r["precision"]) for r in
              (evaluate(data, keys, lambda d, s, c=c: baseline_output(d, c)) for c in np.arange(0.1, 0.91, 0.1))]
        ax.plot(*zip(*pr), "o-", color="#8a96a8", label="프레임별 최고 검출 (conf 0.1~0.9)")
        r = results[(g, "추적(조정된 파라미터)")]
        ax.scatter(r["recall"], r["precision"], marker="*", s=260, color="#1f5fd0", label="추적", zorder=5)
        ax.set_title(g)
        ax.set_xlabel("재현율")
        ax.set_ylabel("정밀도")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/round{ROUND}_precision_recall.png", dpi=160)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    xs = np.arange(0, 61, 1)
    for ax, g in zip(axes, ["video10 (시험)", "TrackNet game6-10 (시험)"]):
        for name in systems:
            color = ("#1f5fd0" if name.startswith("추적(조정") else "#8a96a8" if "1차" in name
                     else "#2a9d8f" if "2차" in name else "#c0392b" if "0.35" in name else "#e9a23b")
            err = results[(g, name)]["err"]
            ax.plot(xs, [np.mean(err <= x) * 100 for x in xs], label=name, color=color)
        ax.set_title(g)
        ax.set_xlabel("허용 오차 (px, 영상 해상도 기준)")
        ax.set_ylabel("공이 보이는 프레임 중 적중 비율 (%)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/round{ROUND}_recall_vs_tolerance.png", dpi=160)
    plt.close()
    print("그래프 저장:", OUT_DIR)


if __name__ == "__main__":
    main()
