# ==============================================================================
# Model Serving API Dockerfile (FastAPI + ONNX Runtime via pyproject.toml)
# ==============================================================================
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy pyproject.toml
COPY pyproject.toml /tmp/pyproject.toml

WORKDIR /app

# Install [api] dependencies directly from pyproject.toml
RUN uv pip install --system --no-cache "/tmp/pyproject.toml[api]"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
