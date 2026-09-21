# src/core/ml_config.py
"""
Central ML Configuration Loader (Single Source of Truth - SSOT)
Parses configs/ml_config.yaml into strongly-typed Pydantic models.
Accessible across Core, Data Engineering, Airflow DAGs, and ML pipelines.

Runtime values (start_date, end_date) are NOT stored here — they are owned by
Airflow dag_run.conf and passed explicitly into pipeline callables.
"""
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

try:
    import yaml
except ImportError:
    yaml = None


class DataConfig(BaseModel):
    mart_table: str = "data_warehouse.mart_demand_prediction"
    timestamp_col: str = "pickup_datetime"
    target_col: str = "total_trips"
    group_cols: List[str] = ["pickup_zone", "pickup_borough", "service_type"]
    group_col: str = "group_id"
    # Fixed pipeline design parameter — not a runtime date.
    test_months: int = 2
    frequency: str = "1h"


class FeaturesConfig(BaseModel):
    lag_hours: List[int] = [24, 168]
    rolling_windows: List[int] = [6, 24]
    use_cyclical_encodings: bool = True
    categorical_features: List[str] = ["time_of_day", "group_id"]


class ModelConfig(BaseModel):
    model_type: str = "LightGBM"
    hyperparameters: Dict[str, Any] = Field(default_factory=lambda: {
        "n_estimators": 1500,
        "learning_rate": 0.05,
        "num_leaves": 128,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "objective": "regression",
        "metric": "mae",
        "importance_type": "gain",
        "verbose": -1,
    })
    early_stopping_rounds: int = 50


class EvidentlyTestsConfig(BaseModel):
    numeric: str = "wasserstein"
    categorical: str = "jensen_shannon"
    target: str = "ks"


class EvidentlyConfig(BaseModel):
    enabled: bool = True
    reference_window_days: int = 60
    current_window_days: int = 14
    drift_share_threshold: float = 0.3
    target_drift_threshold: float = 0.05
    confidence_level: float = 0.95
    tests: EvidentlyTestsConfig = Field(default_factory=EvidentlyTestsConfig)


class MonitoringConfig(BaseModel):
    evidently: EvidentlyConfig = Field(default_factory=EvidentlyConfig)


class ServingConfig(BaseModel):
    export_onnx: bool = True
    onnx_output_path: str = "./models/lightgbm_model.onnx"
    target_opset: int = 15
    parity_tolerance: float = 0.0001
    benchmark_runs: int = 500


class MLflowConfig(BaseModel):
    """
    ML-specific MLflow settings (experiment name, registered model name).
    Infrastructure settings (tracking URI, S3 endpoint) delegate to core.config.MLflowSettings SSOT.
    """
    registered_model_name: str = "taxi-demand-forecaster"
    tracking_uri: Optional[str] = None
    artifact_bucket: Optional[str] = None

    def get_tracking_uri(self) -> str:
        """Resolve tracking URI from core settings SSOT (Docker-aware)."""
        if self.tracking_uri:
            return self.tracking_uri
        from core.config import settings
        return settings.mlflow.resolved_tracking_uri

    def get_experiment_name(self) -> str:
        """Resolve experiment name from core settings SSOT."""
        from core.config import settings
        return settings.mlflow.experiment_name

    def get_artifact_bucket(self) -> str:
        """Resolve artifact bucket from core settings SSOT."""
        if self.artifact_bucket:
            return self.artifact_bucket
        from core.config import settings
        return settings.mlflow.artifact_bucket


class PipelineMetadataConfig(BaseModel):
    name: str = "taxitrack_demand_forecasting_pipeline"
    version: str = "2.0.0"
    environment: str = "production"


class MLConfig(BaseModel):
    """Unified SSOT configuration for TaxiTrack Machine Learning."""
    pipeline: PipelineMetadataConfig = Field(default_factory=PipelineMetadataConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)

    @classmethod
    def from_yaml(cls, filepath: Optional[str] = None) -> "MLConfig":
        """Load configuration from YAML file or return defaults if file not found."""
        if filepath is None:
            candidates = [
                Path("configs/ml_config.yaml"),
                Path("/opt/airflow/configs/ml_config.yaml"),
                Path(__file__).resolve().parent.parent.parent / "configs" / "ml_config.yaml",
            ]
            for c in candidates:
                if c.exists():
                    filepath = str(c)
                    break

        if filepath and os.path.exists(filepath) and yaml is not None:
            with open(filepath, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
                return cls(**raw) if raw else cls()

        return cls()


# Global ML Configuration Singleton
ml_config: MLConfig = MLConfig.from_yaml()
