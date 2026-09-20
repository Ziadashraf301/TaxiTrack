# src/ml/__init__.py
"""
Machine Learning & MLOps Package for TaxiTrack Platform (Modular Clean Architecture)
Provides backward-compatible re-exports from cohesive subpackages:
  - ml.data: Data access & feature repositories
  - ml.features: Temporal feature engineering & lookback buffers
  - ml.models: Forecasting strategies, factory, and evaluators
  - ml.monitoring: Evidently AI multi-type drift observability engine
  - ml.serving: ONNX conversion, parity validation, and latency benchmarking
  - ml.tracking: MLflow experiment tracking and model registry lifecycle
  - ml.graph: Spatial network and transit corridor graph analytics
  - ml.pipeline: MLTrainingPipeline facade
"""

# 1. Data Layer
from ml.data.base import BaseDemandRepository, BaseNetworkRepository
from ml.data.clickhouse import ClickHouseFeatureRepository

# 2. Features Layer
from ml.features.base import BaseFeatureEngineer
from ml.features.temporal import TemporalFeatureEngineer

# 3. Models Layer
from ml.models.base import BaseForecaster, ModelEvaluator
from ml.models.lightgbm import LightGBMForecaster
from ml.models.xgboost import XGBoostForecaster
from ml.models.factory import ForecasterFactory

# 4. Monitoring Layer (Evidently AI)
from ml.monitoring.detector import EvidentlyDriftDetector
from ml.monitoring.pipeline import DriftMonitoringPipeline

# 5. Serving Layer (ONNX)
from ml.serving.onnx import ONNXModelExporter

# 6. Tracking Layer (MLflow)
from ml.tracking.base import BaseExperimentTracker
from ml.tracking.mlflow import MLflowExperimentTracker

# 7. Graph Analytics Layer
from ml.graph.base import BaseGraphAnalyzer
from ml.graph.spatial import SpatialNetworkAnalyzer

# 8. Orchestrator Pipeline
from ml.pipeline import MLTrainingPipeline

__all__ = [
    # Data
    "BaseDemandRepository",
    "BaseNetworkRepository",
    "ClickHouseFeatureRepository",
    # Features
    "BaseFeatureEngineer",
    "TemporalFeatureEngineer",
    # Models
    "BaseForecaster",
    "ModelEvaluator",
    "LightGBMForecaster",
    "XGBoostForecaster",
    "ForecasterFactory",
    # Monitoring
    "EvidentlyDriftDetector",
    "DriftMonitoringPipeline",
    # Serving
    "ONNXModelExporter",
    # Tracking
    "BaseExperimentTracker",
    "MLflowExperimentTracker",
    # Graph
    "BaseGraphAnalyzer",
    "SpatialNetworkAnalyzer",
    # Pipeline
    "MLTrainingPipeline",
]
