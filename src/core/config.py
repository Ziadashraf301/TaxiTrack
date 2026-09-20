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


class MinioSettings(BaseSettings):
    """MinIO S3 Object Storage Configuration"""
    model_config = SettingsConfigDict(env_prefix="MINIO_", extra="ignore")

    root_user: str = Field(default="admin", alias="MINIO_ROOT_USER")
    root_password: str = Field(default="password123", alias="MINIO_ROOT_PASSWORD")
    host: str = Field(default="minio", alias="MINIO_HOST")
    port: int = Field(default=9000, alias="MINIO_PORT")
    console_port: int = Field(default=9001, alias="MINIO_CONSOLE_PORT")
    secure: bool = Field(default=False, alias="MINIO_SECURE")

    def __init__(self, **data):
        if not data and not hasattr(self, "_settings_build_values"):
            data = {
                "root_user": os.getenv("MINIO_ROOT_USER", "admin"),
                "root_password": os.getenv("MINIO_ROOT_PASSWORD", "password123"),
                "host": os.getenv("MINIO_HOST", "minio"),
                "port": int(os.getenv("MINIO_PORT", "9000")),
                "console_port": int(os.getenv("MINIO_CONSOLE_PORT", "9001")),
                "secure": os.getenv("MINIO_SECURE", "false").lower() == "true",
            }
        super().__init__(**data)

    @property
    def endpoint(self) -> str:
        return f"{self.host}:{self.port}"


class ClickHouseSettings(BaseSettings):
    """ClickHouse OLAP Feature Store & Data Warehouse Configuration"""
    model_config = SettingsConfigDict(env_prefix="CLICKHOUSE_", extra="ignore")

    user: str = Field(default="default", alias="CLICKHOUSE_USER")
    password: str = Field(default="", alias="CLICKHOUSE_PASSWORD")
    db: str = Field(default="data_warehouse", alias="CLICKHOUSE_DB")
    host: str = Field(default="clickhouse", alias="CLICKHOUSE_HOST")
    http_port: int = Field(default=8123, alias="CLICKHOUSE_HTTP_PORT")
    tcp_port: int = Field(default=9005, alias="CLICKHOUSE_TCP_PORT")

    def __init__(self, **data):
        if not data and not hasattr(self, "_settings_build_values"):
            data = {
                "user": os.getenv("CLICKHOUSE_USER", "default"),
                "password": os.getenv("CLICKHOUSE_PASSWORD", ""),
                "db": os.getenv("CLICKHOUSE_DB", "data_warehouse"),
                "host": os.getenv("CLICKHOUSE_HOST", "clickhouse"),
                "http_port": int(os.getenv("CLICKHOUSE_HTTP_PORT", "8123")),
                "tcp_port": int(os.getenv("CLICKHOUSE_TCP_PORT", "9005")),
            }
        super().__init__(**data)


class PostgresSettings(BaseSettings):
    """Unified PostgreSQL Metadata Store Configuration"""
    model_config = SettingsConfigDict(env_prefix="POSTGRES_", extra="ignore")

    user: str = Field(default="airflow", alias="POSTGRES_USER")
    password: str = Field(default="airflow", alias="POSTGRES_PASSWORD")
    db: str = Field(default="airflow", alias="POSTGRES_DB")
    host: str = Field(default="postgres", alias="POSTGRES_HOST")
    port: int = Field(default=5432, alias="POSTGRES_PORT")

    def __init__(self, **data):
        if not data and not hasattr(self, "_settings_build_values"):
            data = {
                "user": os.getenv("POSTGRES_USER", "airflow"),
                "password": os.getenv("POSTGRES_PASSWORD", "airflow"),
                "db": os.getenv("POSTGRES_DB", "airflow"),
                "host": os.getenv("POSTGRES_HOST", "postgres"),
                "port": int(os.getenv("POSTGRES_PORT", "5432")),
            }
        super().__init__(**data)


class MLflowSettings(BaseSettings):
    """MLflow Tracking Server Configuration"""
    model_config = SettingsConfigDict(env_prefix="MLFLOW_", extra="ignore")

    tracking_uri: str = Field(default="http://mlflow:5000", alias="MLFLOW_TRACKING_URI")
    experiment_name: str = Field(default="taxitrack_demand_forecasting", alias="MLFLOW_EXPERIMENT_NAME")
    artifact_bucket: str = Field(default="mlflow-artifacts", alias="MLFLOW_ARTIFACT_BUCKET")

    def __init__(self, **data):
        if not data and not hasattr(self, "_settings_build_values"):
            data = {
                "tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
                "experiment_name": os.getenv("MLFLOW_EXPERIMENT_NAME", "taxitrack_demand_forecasting"),
                "artifact_bucket": os.getenv("MLFLOW_ARTIFACT_BUCKET", "mlflow-artifacts"),
            }
        super().__init__(**data)


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

    def __init__(self, **data):
        if not data and not hasattr(self, "_settings_build_values"):
            data = {
                "app_name": "TaxiTrack",
                "environment": os.getenv("ENVIRONMENT", "development"),
                "base_url": os.getenv("NYC_TLC_BASE_URL", "https://d37ci6vzurychx.cloudfront.net/trip-data"),
                "minio": MinioSettings(),
                "clickhouse": ClickHouseSettings(),
                "postgres": PostgresSettings(),
                "mlflow": MLflowSettings(),
            }
        super().__init__(**data)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """
    Singleton accessor for application settings.
    Guarantees thread-safe, cached loading from .env.
    """
    return AppSettings()


# Global singleton instance
settings = get_settings()
