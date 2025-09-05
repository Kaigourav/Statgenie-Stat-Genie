# ---------- STAGE 1: Build ----------
FROM python:3.11-slim as builder

ENV DEBIAN_FRONTEND=noninteractive

# Install build packages and system libraries required by python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    ca-certificates \
    pkg-config \
    libpq-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    libfreetype6-dev \
    libzbar-dev \
    poppler-utils \
    ghostscript \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libx11-6 \
    chromium \
    fonts-liberation \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# ---------- STAGE 2: Runtime ----------
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# Create a non-root user
RUN groupadd -r app && useradd -r -g app -d /app -s /sbin/nologin app \
    && mkdir -p /app/uploads \
    && chown -R app:app /app

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libx11-6 \
    chromium \
    poppler-utils \
    ghostscript \
    tesseract-ocr \
    fonts-liberation \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app /app

ENV PYTHONUNBUFFERED=1

# Fix env var for Kaleido
ENV KALENVAUTO=1
ENV KALIEDO_BROWSER_PATH=/usr/bin/chromium

ARG PORT=8080
EXPOSE ${PORT}

USER app

# Gunicorn entrypoint
CMD ["gunicorn", "--bind", ":8080", "--workers=1", "--threads=2", "--timeout=300", "app:app"]
