FROM python:3.11-slim

WORKDIR /app

# NOTE: libgl1-mesa-glx was removed in Debian trixie (current python:3.11-slim
# base); libgl1 is its replacement and provides the same libGL.so.1 for OpenCV.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /data && chmod 777 /data

COPY backend/requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]