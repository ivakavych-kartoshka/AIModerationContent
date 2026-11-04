FROM python:3.11-slim-bookworm

# Env
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONIOENCODING=utf-8 \
    HF_HOME=/home/user/.cache/huggingface

# Install minimal system deps + pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -u 1000 user \
    && mkdir -p /app \
    && chown -R user:user /app

WORKDIR /app

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Copy only needed source files (src/ and main.py) + config
COPY --chown=user:user src ./src
COPY --chown=user:user main.py .

# Minimal requirements for moderation API:
# - fastapi + uvicorn for the server
# - pydantic for request validation
# - langid for language detection (fallback path; hybrid uses sklearn etc. but will gracefully fallback)
# - PyYAML for config loading
# - scipy / numpy optional (heuristic uses numpy? fallback only uses built-in)
RUN pip install --no-cache-dir \
    fastapi==0.110.0 \
    uvicorn[standard]==0.29.0 \
    pydantic==2.7.4 \
    langid==1.1.6 \
    PyYAML==6.0.1

# Expose FastAPI moderation service port
EXPOSE 8001

# Healthcheck based on the /health endpoint exposed by main.py
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${PORT}/health || exit 1

# Port mặc định cho moderation service (NestJS AI_MODERATION_URL đang dùng .../api/v1)
ENV PORT=8001

# 1 worker để không nhân đôi RAM. Bind 0.0.0.0 để gateway/proxy (docker, HF, Railway) truy cập được.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1"]