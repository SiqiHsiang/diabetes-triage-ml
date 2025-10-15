# ---------- Build stage: install deps + train the model ----------
FROM python:3.11-slim AS builder
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# System deps (curl for healthcheck later, build tools minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and train inside the image so the model is self-contained
COPY src/ src/
ARG MODEL_KIND=linear
ENV MODEL_KIND=${MODEL_KIND}
RUN echo ">> MODEL_KIND=${MODEL_KIND}" && python -m src.train --out_dir models --kind ${MODEL_KIND}

# ---------- Runtime stage: lightweight server image ----------
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MODEL_DIR=/app/models
WORKDIR /app

# Bring in site-packages and our app + trained artifacts
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/src /app/src
COPY --from=builder /app/models /app/models

EXPOSE 8000

# Optional healthcheck so orchestrators know it's alive
HEALTHCHECK --interval=30s --timeout=3s --retries=3 CMD curl -sf http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]