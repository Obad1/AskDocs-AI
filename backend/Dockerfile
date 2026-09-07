# AskDocs AI backend — local-first, zero-API-key, zero-login.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for pdfium / tesseract / audio build tooling.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Persistence volume mounted at /data by docker-compose.
RUN mkdir -p /data/chroma /data/exports

EXPOSE 8000

# Huey consumer is run separately (e.g. huey_consumer app.core.pipeline.huey).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
