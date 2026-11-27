"""공 추적: 프레임별 공 후보(x, y, conf)를 움직임이 일관된 경로로 이어 붙인다. 학습은 필요 없다.

방식(영상 전체를 한 번에 처리하는 오프라인 방식)
- 각 후보에 점수(conf 의 로짓 + 보정)를 주고, 프레임 사이 이동 속도가 현실적일수록 비용이 낮은 연결로 본다.
- 동적 계획법으로 "점수 합 - 연결 비용 - 구간 시작 비용"이 최대인 경로(여러 구간 가능)를 찾는다.
  구간 시작 비용 때문에 고립된 오검출은 채택되지 않고, 이어지는 후보들만 남는다.
- 후보가 없는 프레임은 최대 max_gap 장까지 건너뛰어 이을 수 있고, 건너뛴 구간은 직선으로 채운다(보간).

출력: (T, 4) 배열 [x, y, conf, 상태]. 상태 0 = 공 없음, 1 = 검출, 2 = 보간.
"""
from dataclasses import dataclass

import numpy as np

NONE, DETECTED, INTERPOLATED = 0, 1, 2


@dataclass
class TrackerConfig:
    max_gap: int = 8             # 후보가 없는 프레임을 최대 몇 장까지 건너뛰어 이을지
    max_speed: float = 0.12      # 프레임당 최대 이동(화면 너비 대비). 넘으면 같은 공으로 보지 않음
    start_cost: float = 4.0      # 새 구간을 시작하는 비용(고립된 오검출을 막음)
    conf_bias: float = 2.2       # 점수 = logit(conf) + conf_bias (conf 0.1 에서 0)
    dist_weight: float = 3.0     # 이동 속도 비용 가중치
    gap_cost: float = 0.4        # 건너뛴 프레임당 비용
    still_speed: float = 0.002   # 이보다 느리면 "정지한 물체(신발 등)"로 보고 비용을 더함(프레임당, 화면 너비 대비)
    still_cost: float = 1.0
    min_nodes: int = 3           # 구간에 포함된 검출이 이보다 적으면 버림
    min_peak_conf: float = 0.0   # 구간 안에 이 점수 이상의 확실한 검출이 하나도 없으면 버림(0 이면 사용 안 함)
    trim_conf: float = 0.0       # 구간의 앞·뒤 끝에서 이 점수보다 낮은 검출을 잘라냄(공이 사라진 뒤 오검출로 구간이 늘어나는 것을 방지, 0 이면 사용 안 함)


def _score(conf):
    c = np.clip(conf, 0.01, 0.99)
    return np.log(c / (1 - c))


def track_ball(dets, width, cfg=None):
    cfg = cfg or TrackerConfig()
    T = len(dets)
    out = np.zeros((T, 4), np.float32)
    if T == 0:
        return out
    pts = [np.asarray(d, dtype=np.float64).reshape(-1, 3) for d in dets]
    reward = [_score(p[:, 2]) + cfg.conf_bias for p in pts]
    f = [np.full(len(p), -np.inf) for p in pts]
    prev = [[None] * len(p) for p in pts]
    best_val = np.zeros(T)
    best_arg = [None] * T

    for t in range(T):
        gv = best_val[t - 1] if t > 0 else 0.0
        ga = best_arg[t - 1] if t > 0 else None
        for i in range(len(pts[t])):
            score, link = gv - cfg.start_cost, ("S", ga)
            for g in range(1, cfg.max_gap + 2):
                t0 = t - g
                if t0 < 0:
                    break
                if len(pts[t0]) == 0:
                    continue
                dist = np.hypot(pts[t0][:, 0] - pts[t][i, 0], pts[t0][:, 1] - pts[t][i, 1]) / width
                speed = dist / g
                ok = speed <= cfg.max_speed
                if not ok.any():
                    continue
                cost = (cfg.dist_weight * (speed / cfg.max_speed) ** 2 + cfg.gap_cost * (g - 1)
                        + cfg.still_cost * np.clip(1 - speed / cfg.still_speed, 0, 1))
                cand = np.where(ok, f[t0] - cost, -np.inf)
                j = int(np.argmax(cand))
                if cand[j] > score:
                    score, link = cand[j], (t0, j)
            f[t][i] = score + reward[t][i]
            prev[t][i] = link
        bv, ba = gv, ga
        for i in range(len(pts[t])):
            if f[t][i] > bv:
                bv, ba = f[t][i], (t, i)
        best_val[t], best_arg[t] = bv, ba

    chains, chain, node = [], [], best_arg[T - 1]
    while node is not None:
        chain.append(node)
        link = prev[node[0]][node[1]]
        if link[0] == "S":
            chains.append(chain[::-1])
            chain, node = [], link[1]
        else:
            node = link
    chains = chains[::-1]

    for chain in chains:
        if cfg.trim_conf > 0:
            while chain and pts[chain[0][0]][chain[0][1], 2] < cfg.trim_conf:
                chain = chain[1:]
            while chain and pts[chain[-1][0]][chain[-1][1], 2] < cfg.trim_conf:
                chain = chain[:-1]
        if len(chain) < cfg.min_nodes:
            continue
        if cfg.min_peak_conf > 0 and max(pts[t][i, 2] for t, i in chain) < cfg.min_peak_conf:
            continue
        for (t, i), nxt in zip(chain, chain[1:] + [None]):
            x, y, c = pts[t][i]
            out[t] = (x, y, c, DETECTED)
            if nxt is not None and nxt[0] - t > 1:
                x2, y2, _ = pts[nxt[0]][nxt[1]]
                for k in range(t + 1, nxt[0]):
                    a = (k - t) / (nxt[0] - t)
                    out[k] = (x + (x2 - x) * a, y + (y2 - y) * a, 0.0, INTERPOLATED)
    return out
