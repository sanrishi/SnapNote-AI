FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    xz-utils \
    && rm -rf /var/lib/apt/lists/*

# Typst 0.15.1 — exact version validated in the composition benchmark.
# Official static musl binary (no runtime deps). The trailing `typst --version`
# fails the Docker build loudly if the binary is missing or broken, so a
# production image can never silently lack Typst (else v3 would degrade to
# bare-hero fallback without anyone noticing).
ARG TYPST_VERSION=0.15.1
RUN curl -fsSL -o /tmp/typst.tar.xz \
      https://github.com/typst/typst/releases/download/v${TYPST_VERSION}/typst-x86_64-unknown-linux-musl.tar.xz \
    && tar -xJf /tmp/typst.tar.xz -C /tmp \
    && mv /tmp/typst-x86_64-unknown-linux-musl/typst /usr/local/bin/typst \
    && rm -rf /tmp/typst.tar.xz /tmp/typst-x86_64-unknown-linux-musl \
    && typst --version

RUN mkdir -p /data && chmod 777 /data

COPY backend/requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]