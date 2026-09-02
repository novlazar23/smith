# =============================================================================
# Trading Orchestra — API Service (FastAPI)
# =============================================================================

# ---------- Build stage ----------
FROM python:3.12-slim AS base

# Avoid .pyc files and prompts
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---------- Dependencies ----------
FROM base AS deps

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml README.md ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -e ".[paper]"

# ---------- Production ----------
FROM base AS production

# System deps (curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages
COPY --from=deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy application code
COPY pyproject.toml README.md ./
COPY packages/ ./packages/
COPY apps/ ./apps/

# Artefakt-Verzeichnis des Backtests (named volume initialisiert es mit
# appuser-Besitz — root-gebundene Bind-Mounts wären für appuser nicht schreibbar)
RUN mkdir -p /app/backtest_reports

# Copy secret-aware entrypoint (before chown below so appuser owns it)
COPY apps/api/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Create non-root user (Privilege-Drop erfolgt in /app/entrypoint.sh via setpriv)
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser && \
    chown -R appuser:appuser /app

# Expose internal API port
EXPOSE 8000

# Healthcheck (docker-level)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production entrypoint — uvicorn with workers
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "apps.api.main:create_app", "--factory", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
