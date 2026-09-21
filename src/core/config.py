# src/core/config.py
"""
Centralized Application Configuration (Singleton Pattern)
Parses environment variables and .env file using Pydantic Settings.
"""
import os
from functools import lru_cache
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv("/opt/airflow/.env")
except ImportError:
    pass

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    class BaseSettings(BaseModel):
        """Fallback when pydantic-settings is not installed in the active environment."""
        pass

    def SettingsConfigDict(**kwargs):
        return kwargs


def is_in_docker() -> bool:
    """Detect if executing inside a Docker container or Airflow worker."""
    return os.path.exists("/.dockerenv") or bool(os.getenv("AIRFLOW_HOME"))


def _resolve_host(env_var: str, current_value: str, docker_hostname: str) -> str:
    """
    Shared Docker-aware host resolution logic.
    Priority: explicit env var > non-localhost value > docker/localhost based on runtime.
    """
    env_host = os.getenv(env_var)
    if env_host:
        return env_host
    if current_value and current_value not in ("localhost", "127.0.0.1"):
        return current_value
    return docker_hostname if is_in_docker() else "localhost"


class MinioSettings(BaseSettings):
    """MinIO S3 Object Storage Configuration (Single Source of Truth)"""
    model_config = SettingsConfigDict(env_prefix="MINIO_", extra="ignore")

    root_user: str = Field(default="admin", alias="MINIO_ROOT_USER")
    root_password: str = Field(default="password123", alias="MINIO_ROOT_PASSWORD")
    host: str = Field(default="localhost", alias="MINIO_HOST")
    port: int = Field(default=9000, alias="MINIO_PORT")
    console_port: int = Field(default=9001, alias="MINIO_CONSOLE_PORT")
    secure: bool = Field(default=False, alias="MINIO_SECURE")

    @property
    def resolved_host(self) -> str:
        return _resolve_host("MINIO_HOST", self.host, "minio")

    @property
    def endpoint(self) -> str:
        return f"{self.resolved_host}:{self.port}"

    @property
    def endpoint_url(self) -> str:
        scheme = "https" if self.secure else "http"
        return f"{scheme}://{self.endpoint}"


class ClickHouseSettings(BaseSettings):
    """ClickHouse OLAP Feature Store & Data Warehouse Configuration"""
    model_config = SettingsConfigDict(env_prefix="CLICKHOUSE_", extra="ignore")

    user: str = Field(default="default", alias="CLICKHOUSE_USER")
    password: str = Field(default="", alias="CLICKHOUSE_PASSWORD")
    db: str = Field(default="data_warehouse", alias="CLICKHOUSE_DB")
    host: str = Field(default="localhost", alias="CLICKHOUSE_HOST")
    http_port: int = Field(default=8123, alias="CLICKHOUSE_HTTP_PORT")
    tcp_port: int = Field(default=9005, alias="CLICKHOUSE_TCP_PORT")

    @property
    def resolved_host(self) -> str:
        return _resolve_host("CLICKHOUSE_HOST", self.host, "clickhouse")


class PostgresSettings(BaseSettings):
    """Unified PostgreSQL Metadata Store Configuration"""
    model_config = SettingsConfigDict(env_prefix="POSTGRES_", extra="ignore")

    user: str = Field(default="airflow", alias="POSTGRES_USER")
    password: str = Field(default="airflow", alias="POSTGRES_PASSWORD")
    db: str = Field(default="airflow", alias="POSTGRES_DB")
    host: str = Field(default="localhost", alias="POSTGRES_HOST")
    port: int = Field(default=5432, alias="POSTGRES_PORT")

    @property
    def resolved_host(self) -> str:
        return _resolve_host("POSTGRES_HOST", self.host, "postgres")


class MLflowSettings(BaseSettings):
    """
    MLflow Tracking Server Infrastructure Configuration (Single Source of Truth).
    Owns: tracking_uri, experiment_name, artifact_bucket, S3 endpoint.
    ML-specific settings (registered_model_name, hyperparameters) live in ml_config.yaml.
    """
    model_config = SettingsConfigDict(env_prefix="MLFLOW_", extra="ignore")

    tracking_uri: str = Field(default="http://localhost:5000", alias="MLFLOW_TRACKING_URI")
    experiment_name: str = Field(default="taxitrack_demand_forecasting", alias="MLFLOW_EXPERIMENT_NAME")
    artifact_bucket: str = Field(default="mlflow-artifacts", alias="MLFLOW_ARTIFACT_BUCKET")

    @property
    def resolved_tracking_uri(self) -> str:
        raw = os.getenv("MLFLOW_TRACKING_URI") or self.tracking_uri
        # Inside Docker: if targeting localhost, route to container network 'mlflow'
        if is_in_docker() and ("localhost" in raw or "127.0.0.1" in raw):
            return "http://mlflow:5000"
        # Outside Docker (host): if target has docker container hostname 'mlflow', route to localhost
        if not is_in_docker() and "mlflow:" in raw:
            return "http://localhost:5000"
        return raw

    @property
    def s3_endpoint_url(self) -> str:
        """MinIO S3 endpoint for MLflow artifacts - derived from MinioSettings SSOT."""
        endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL")
        if endpoint:
            if not is_in_docker() and "minio:" in endpoint:
                return "http://localhost:9000"
            return endpoint
        # Fallback to MinIO host resolution
        minio_host = os.getenv("MINIO_HOST") or ("minio" if is_in_docker() else "localhost")
        minio_port = os.getenv("MINIO_PORT", "9000")
        return f"http://{minio_host}:{minio_port}"


class AppSettings(BaseSettings):
    """Global Application Settings Aggregator"""
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "TaxiTrack"
    environment: str = Field(default="development", alias="ENVIRONMENT")
    base_url: str = Field(
        default="https://d37ci6vzurychx.cloudfront.net/trip-data",
        alias="NYC_TLC_BASE_URL"
    )

    minio: MinioSettings = Field(default_factory=MinioSettings)
    clickhouse: ClickHouseSettings = Field(default_factory=ClickHouseSettings)
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    mlflow: MLflowSettings = Field(default_factory=MLflowSettings)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """
    Singleton accessor for application settings.
    Guarantees thread-safe, cached loading from .env.
    """
    return AppSettings()


# Global singleton instance
settings = get_settings()
