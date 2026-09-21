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
try:
    from ml.data.base import BaseDemandRepository, BaseNetworkRepository
    from ml.data.clickhouse import ClickHouseFeatureRepository
except ImportError:
    pass

# 2. Features Layer
try:
    from ml.features.base import BaseFeatureEngineer
    from ml.features.temporal import TemporalFeatureEngineer
except ImportError:
    pass

# 3. Models Layer
try:
    from ml.models.base import BaseForecaster, ModelEvaluator
    from ml.models.lightgbm import LightGBMForecaster
    from ml.models.xgboost import XGBoostForecaster
    from ml.models.factory import ForecasterFactory
except ImportError:
    pass

# 4. Monitoring Layer (Evidently AI)
try:
    from ml.monitoring.detector import EvidentlyDriftDetector
    from ml.monitoring.pipeline import DriftMonitoringPipeline
except ImportError:
    pass

# 5. Serving Layer (ONNX)
try:
    from ml.serving.onnx import ONNXModelExporter
except ImportError:
    pass

# 6. Tracking Layer (MLflow)
try:
    from ml.tracking.base import BaseExperimentTracker
    from ml.tracking.mlflow import MLflowExperimentTracker
except ImportError:
    pass

# 7. Graph Analytics Layer
try:
    from ml.graph.base import BaseGraphAnalyzer
    from ml.graph.spatial import SpatialNetworkAnalyzer
except ImportError:
    pass

# 8. Orchestrator Pipeline
try:
    from ml.pipeline import MLTrainingPipeline
except ImportError:
    pass

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
