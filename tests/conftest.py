"""테스트 공통 설정.

- 서비스 코드는 수정하지 않고, 테스트에서만 환경을 바꾼다.
- pipeline.py 는 import 시점에 YOLO 모델을 불러오므로, 기본값으로는 ultralytics 를 가짜 모듈로 대체한다
  (모델 다운로드 없이 검출 이외의 로직을 검증). 실제 모델로 돌리려면 CV_REAL_YOLO=1.
"""
import os
import sys
import tempfile
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "webapp" / "backend"
sys.path.insert(0, str(BACKEND))

_TMP = tempfile.mkdtemp(prefix="cv_test_")
os.environ["COURTVISION_DB"] = os.path.join(_TMP, "import_time.db")
os.environ["UPLOAD_DIR"] = os.path.join(_TMP, "uploads")
os.environ["RESULTS_DIR"] = os.path.join(_TMP, "results")

if os.environ.get("CV_REAL_YOLO") != "1":
    fake = types.ModuleType("ultralytics")

    class _FakeYOLO:
        def __init__(self, *args, **kwargs):
            pass

    fake.YOLO = _FakeYOLO
    sys.modules["ultralytics"] = fake


@pytest.fixture
def db(tmp_path, monkeypatch):
    """테스트마다 새 SQLite 파일을 쓰도록 store/accounts 의 DB 경로를 바꾼다."""
    from crawler import store
    import accounts

    path = tmp_path / "test.db"
    monkeypatch.setattr(store, "DB_PATH", path)
    monkeypatch.setattr(accounts, "DB_PATH", path)
    accounts.init()
    return path


@pytest.fixture
def make_video(tmp_path):
    """검은 화면의 짧은 mp4 를 만들어 경로를 돌려주는 함수."""
    def _make(name="clip.mp4", frames=5, size=(1920, 1080), fps=30):
        path = tmp_path / name
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not writer.isOpened():
            pytest.skip("이 환경의 OpenCV 에서 mp4 쓰기를 지원하지 않음")
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        for _ in range(frames):
            writer.write(frame)
        writer.release()
        return str(path)
    return _make
