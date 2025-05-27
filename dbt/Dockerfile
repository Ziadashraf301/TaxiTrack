# dbt/Dockerfile
FROM ghcr.io/dbt-labs/dbt-postgres:1.9.0

# Copy dbt project and profiles
COPY ./dbt /dbt
COPY ./dbt/profiles.yml /root/.dbt/profiles.yml

WORKDIR /dbt

# Install dbt dependencies
RUN dbt deps
