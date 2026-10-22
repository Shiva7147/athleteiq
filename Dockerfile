# Multi-stage production Dockerfile for AthleteIQ 2.0
FROM python:3.12-slim AS builder

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install uv

# Copy dependency files
COPY pyproject.toml ./

# Create virtual environment and install dependencies
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv pip install .[ml,agent]

# ----------------- Final Lightweight Runtime -----------------
FROM python:3.12-slim AS runner

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy source code and evals
COPY core/ ./core/
COPY simulator/ ./simulator/
COPY features/ ./features/
COPY ml/ ./ml/
COPY rag/ ./rag/
COPY agent/ ./agent/
COPY multimodal/ ./multimodal/
COPY api/ ./api/
COPY evals/ ./evals/

EXPOSE 8000

# Healthcheck probe using liveness endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/v2/health || exit 1

CMD ["uvicorn", "api.routes:app", "--host", "0.0.0.0", "--port", "8000"]
