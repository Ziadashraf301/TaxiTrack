# ==============================================================================
# V2: Streamlit Dashboard UI (Pure Frontend Client)
# ==============================================================================
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy pyproject.toml as single source of truth
COPY pyproject.toml /tmp/pyproject.toml

WORKDIR /app

# Strictly UI dependencies only (Streamlit, Altair, Pandas, Requests)
RUN uv pip install --system --no-cache "/tmp/pyproject.toml[ui]"

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "demand_prediction/streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
