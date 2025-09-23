# dbt/Dockerfile
FROM ghcr.io/dbt-labs/dbt-postgres:1.9.0

# Install Python packages for ClickHouse + MinIO
RUN pip install --no-cache-dir \
    clickhouse-connect \
    dbt-clickhouse


# Copy dbt project and profiles
COPY ./dbt /dbt
COPY ./dbt/profiles.yml /root/.dbt/profiles.yml

WORKDIR /dbt

# Install dbt dependencies
RUN dbt deps
