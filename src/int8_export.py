"""FP32 / INT8(OpenVINO) 모델 내보내기.
- 대상: 서비스에서 실제 쓰는 yolo11n.pt (선수 검출), imgsz=960
- 캘리브레이션: video2에서 균등 간격 300프레임 (평가에 쓸 video9와 분리)
- INT8: Ultralytics export(int8=True) -> NNCF post-training quantization (재학습 없음)"""
import os
import cv2
from ultralytics import YOLO

IMGSZ = [544, 960]   # 16:9 영상의 PyTorch 직사각형 입력(960x544)과 동일하게 고정
CALIB_VIDEO = "data/raw/archive/video2.mp4"
CALIB_DIR = os.path.abspath("data/int8_calib")
N_CALIB = 300

os.makedirs(f"{CALIB_DIR}/images", exist_ok=True)
os.makedirs(f"{CALIB_DIR}/labels", exist_ok=True)
cap = cv2.VideoCapture(CALIB_VIDEO)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
step = max(total // N_CALIB, 1)
saved = 0
for i in range(0, total, step):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, frame = cap.read()
    if not ok or saved >= N_CALIB:
        continue
    cv2.imwrite(f"{CALIB_DIR}/images/c{i:05d}.jpg", frame)
    open(f"{CALIB_DIR}/labels/c{i:05d}.txt", "w").close()
    saved += 1
cap.release()
print(f"캘리브레이션 프레임 {saved}장 (video2, 간격 {step})")

with open(f"{CALIB_DIR}/data.yaml", "w") as f:
    f.write(f"path: {CALIB_DIR}\ntrain: images\nval: images\nnames:\n  0: person\n")

os.chdir("models/int8")   # 1차 시도는 imgsz=960(정사각형 고정)이라 PyTorch(직사각형)와 조건이 달라 폐기, 결과는 *_sq960 로 보관
# FP32 (변환만, 양자화 없음)
YOLO("yolo11n.pt").export(format="openvino", imgsz=IMGSZ, half=False)
os.rename("yolo11n_openvino_model", "yolo11n_fp32_openvino_model")
# INT8
YOLO("yolo11n.pt").export(format="openvino", imgsz=IMGSZ, int8=True, data=f"{CALIB_DIR}/data.yaml")
os.rename("yolo11n_int8_openvino_model", "yolo11n_int8_openvino_model") if os.path.isdir("yolo11n_int8_openvino_model") else None
print(sorted(os.listdir(".")))
