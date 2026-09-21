# ==============================================================================
# Airflow Dockerfile (Single Source of Truth: pyproject.toml via Astral UV)
# ==============================================================================
FROM apache/airflow:2.10.2

USER root

# Install system dependencies (libgomp1 is required by LightGBM for OpenMP support)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Increase network timeout for large wheels (prevents 30s timeout failures)
ENV UV_HTTP_TIMEOUT=180

# Install uv binary from official Astral image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy pyproject.toml as the single source of truth
COPY pyproject.toml /tmp/pyproject.toml

# Compile and install dependencies directly from pyproject.toml using uv
# --system ensures uv inspects pre-installed packages in apache/airflow:2.10.2 to avoid redundant downloads
RUN uv pip compile --system /tmp/pyproject.toml --extra data --extra ml --extra monitoring -o /tmp/reqs.tmp \
    && uv pip install --system --no-cache -r /tmp/reqs.tmp \
    && rm -f /tmp/reqs.tmp

USER airflow
WORKDIR /opt/airflow



