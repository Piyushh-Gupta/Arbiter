FROM python:3.12-slim as builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VERSION=1.8.3 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_NO_INTERACTION=1

ENV PATH="$POETRY_HOME/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends curl build-essential && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
# Only install main dependencies to cache the layer
RUN poetry install --without dev,test --no-root

COPY src/ ./src/
COPY configs/ ./configs/
COPY README.md ./
RUN poetry install --without dev,test

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app
RUN useradd -m -r -u 1000 appuser && chown appuser:appuser /app
COPY --from=builder --chown=appuser:appuser /app /app

USER appuser
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
