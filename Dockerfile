FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib

# OpenCV가 필요로 하는 시스템 라이브러리
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 무겁고 잘 안 바뀌는 것(PyTorch CPU 버전)을 먼저 설치 -> 코드만 바뀌면 이 층은 캐시를 재사용
RUN pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu

COPY webapp/backend/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY webapp /app/webapp

# 일반 사용자로 실행 (root 권한을 주지 않음)
RUN useradd --create-home app \
    && mkdir -p /app/data \
    && chown -R app:app /app
USER app

WORKDIR /app/webapp/backend
EXPOSE 8000

# 세션을 서버 메모리에 두므로 워커는 1개로 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
