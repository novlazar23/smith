FROM ghcr.io/astral-sh/uv:0.11.27 AS uv
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=uv /uv /uvx /bin/
COPY pyproject.toml uv.lock README.md /app/
COPY src /app/src
COPY config /app/config
COPY prompts /app/prompts
COPY schemas /app/schemas

RUN uv sync --frozen --no-dev --no-editable

EXPOSE 8080

CMD ["/app/.venv/bin/uvicorn", "trading_harness.main:app", "--host", "0.0.0.0", "--port", "8080"]
