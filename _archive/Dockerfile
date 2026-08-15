FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install torch==2.12.0 torchvision==0.27.0 \
    --index-url https://download.pytorch.org/whl/cu130

COPY requirements_zimage.txt .
RUN pip install -r requirements_zimage.txt

RUN pip install git+https://github.com/huggingface/diffusers

COPY app.py config.py server.py ./
COPY static ./static

RUN mkdir -p /app/output

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=900s --retries=5 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
