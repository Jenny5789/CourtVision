"""FP32(PyTorch) vs FP32(OpenVINO) vs INT8(OpenVINO) 비교. 평가 영상: video9 (캘리브레이션은 video2, 분리됨)
지표: 크기, 프레임당 시간, FPS, FP32 대비 검출 일치율, BoT-SORT 추적 유지."""
import glob, os, platform, statistics, sys, time
import cv2
import numpy as np
from ultralytics import YOLO

VIDEO = "data/raw/archive/video9.mp4"
N = 300
WARM = 10
IMGSZ = 960
CONF = 0.3
CONFIGS = [
    # PyTorch는 16:9 영상을 960x544로 처리하므로 OpenVINO도 같은 입력(544x960)으로 내보내 조건을 맞춤
    ("PyTorch FP32 (CPU)", "models/int8/yolo11n.pt", "cpu", 960),
    ("OpenVINO FP32 (CPU)", "models/int8/yolo11n_fp32_openvino_model", "intel:cpu", [544, 960]),
    ("OpenVINO INT8 (CPU)", "models/int8/yolo11n_int8_openvino_model", "intel:cpu", [544, 960]),
]
if len(sys.argv) > 1 and sys.argv[1] == "gpu":
    CONFIGS = [("PyTorch FP32 (CUDA)", "models/int8/yolo11n.pt", 0, 960)]

cap = cv2.VideoCapture(VIDEO)
frames = []
while len(frames) < N + WARM:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
cap.release()
print(f"영상 {VIDEO}, 프레임 {len(frames)}(워밍업 {WARM} 포함), {frames[0].shape[1]}x{frames[0].shape[0]}, imgsz={IMGSZ}, conf={CONF}")
print("CPU:", platform.processor())


def size_mb(p):
    if os.path.isdir(p):
        return sum(os.path.getsize(f) for f in glob.glob(p + "/**/*", recursive=True) if os.path.isfile(f)) / 1e6
    return os.path.getsize(p) / 1e6


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / u if u > 0 else 0


def detect_run(model, device, imgsz):
    tot, inf, dets = [], [], []
    for i, fr in enumerate(frames):
        t = time.perf_counter()
        r = model.predict(fr, classes=[0], conf=CONF, imgsz=imgsz, device=device, verbose=False)[0]
        dt = (time.perf_counter() - t) * 1000
        if i >= WARM:
            tot.append(dt); inf.append(r.speed["inference"])
            dets.append([(b.xyxy[0].tolist(), float(b.conf)) for b in r.boxes])
    return tot, inf, dets


def track_run(model, device, imgsz):
    ids = {}
    for i, fr in enumerate(frames[WARM:]):
        r = model.track(fr, classes=[0], conf=CONF, imgsz=imgsz, device=device, tracker="botsort.yaml", persist=True, verbose=False)[0]
        if r.boxes.id is not None:
            for tid in r.boxes.id.int().tolist():
                ids[tid] = ids.get(tid, 0) + 1
    top2 = sorted(ids.values(), reverse=True)[:2]
    return len(ids), top2


ref = None
rows = []
for name, path, dev, isz in CONFIGS:
    m = YOLO(path, task="detect")
    tot, inf, dets = detect_run(m, dev, isz)
    row = dict(name=name, size=size_mb(path), ms=statistics.mean(tot), med=statistics.median(tot),
               p95=float(np.percentile(tot, 95)), inf=statistics.mean(inf), fps=1000 / statistics.mean(tot))
    if ref is None:
        ref = dets
        row.update(rec=1.0, prec=1.0, miou=1.0)
    else:
        matched = nref = nvar = 0; ious = []
        for R, V in zip(ref, dets):
            nref += len(R); nvar += len(V); used = set()
            for rb, _ in R:
                best, bj = 0, None
                for j, (vb, _) in enumerate(V):
                    if j in used: continue
                    v = iou(rb, vb)
                    if v > best: best, bj = v, j
                if best >= 0.5:
                    matched += 1; used.add(bj); ious.append(best)
        row.update(rec=matched / max(nref, 1), prec=matched / max(nvar, 1), miou=float(np.mean(ious)) if ious else 0)
    row["n_ids"], row["top2"] = track_run(YOLO(path, task="detect"), dev, isz)
    rows.append(row)
    print(row, flush=True)

print("\n| 모델 | 크기(MB) | 평균 ms/프레임 | 중앙값 | p95 | FPS | FP32 대비 검출 재현율 | 정밀도 | 평균 IoU | 추적 ID 수 | 상위2 트랙 길이 |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['name']} | {r['size']:.1f} | {r['ms']:.1f} | {r['med']:.1f} | {r['p95']:.1f} | {r['fps']:.1f} | {r['rec']*100:.1f}% | {r['prec']*100:.1f}% | {r['miou']:.3f} | {r['n_ids']} | {r['top2']} |")
