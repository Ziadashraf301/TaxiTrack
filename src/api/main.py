"""TaxiTrack FastAPI Application Entrypoint.

Configures lifespan management, MLflow model loading, Prometheus metrics, and routing.
"""
from contextlib import asynccontextmanager
import os
from typing import AsyncGenerator
from cachetools import TTLCache
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from api.metrics import DRIFT_ALERT_GAUGE, MODEL_VERSION_INFO
from api.routers import analytics_router, forecast_router, network_router
from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager for startup loading and shutdown cleanup."""
    logger.info("Initializing TaxiTrack Serving Layer...")

    # 1. In-process cache initialization
    app.state.cache = TTLCache(maxsize=1024, ttl=300)

    # 2. Attempt to load production model and feature engineer from MLflow Model Registry
    onnx_session = None
    feature_engineer = None
    model_version = "v1.0.0"
    loaded_from_mlflow = False

    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        import onnxruntime as rt
        from ml.features.temporal import TemporalFeatureEngineer

        mlflow_uri = settings.mlflow.resolved_tracking_uri
        logger.info(f"Checking MLflow Model Registry at {mlflow_uri}...")
        client = MlflowClient(tracking_uri=mlflow_uri)

        model_name = "taxi-demand-forecaster"
        target_version = None

        # A. Try modern MLflow aliases (@champion, @production)
        for alias in ("champion", "production"):
            try:
                target_version = client.get_model_version_by_alias(model_name, alias)
                if target_version:
                    logger.info(f"Found model version {target_version.version} via alias '@{alias}'.")
                    break
            except Exception:
                pass

        # B. Fall back to legacy 'Production' stage (suppressing deprecated FutureWarning)
        if not target_version:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                try:
                    stage_versions = client.get_latest_versions(model_name, stages=["Production"])
                    if stage_versions:
                        target_version = stage_versions[0]
                        logger.info(f"Found Production stage model version {target_version.version}.")
                except Exception:
                    pass

        # C. Fall back to latest registered version in registry
        if not target_version:
            try:
                all_versions = client.search_model_versions(
                    f"name = '{model_name}'",
                    order_by=["version_number DESC"],
                    max_results=1,
                )
                if all_versions:
                    target_version = all_versions[0]
                    logger.info(f"Found latest registered model version {target_version.version} (stage: {target_version.current_stage}).")
            except Exception as search_err:
                logger.debug(f"Could not search model versions in MLflow: {search_err}")

        if target_version:
            run_id = target_version.run_id
            version_str = str(target_version.version)
            logger.info(f"Found MLflow model version {version_str} (run_id: {run_id}). Downloading artifacts...")

            os.makedirs("/tmp/mlflow", exist_ok=True)
            local_onnx = None
            for onnx_path in ("onnx/lightgbm_model.onnx", "model_artifacts/lightgbm_model.onnx", "lightgbm_model.onnx"):
                try:
                    local_onnx = client.download_artifacts(run_id, onnx_path, "/tmp/mlflow")
                    if os.path.exists(local_onnx):
                        break
                except Exception:
                    continue

            local_fe = None
            for fe_path in ("feature_engineering/feature_engineer.pkl", "model_artifacts/feature_engineer.pkl", "encoders/feature_engineer.pkl", "feature_engineer.pkl"):
                try:
                    local_fe = client.download_artifacts(run_id, fe_path, "/tmp/mlflow")
                    if os.path.exists(local_fe):
                        break
                except Exception:
                    continue

            if local_onnx and os.path.exists(local_onnx):
                onnx_session = rt.InferenceSession(local_onnx, providers=["CPUExecutionProvider"])
                if local_fe and os.path.exists(local_fe):
                    feature_engineer = TemporalFeatureEngineer().load(local_fe)
                model_version = f"mlflow-v{version_str}"
                loaded_from_mlflow = True
                logger.info(f"Successfully loaded production model {model_version} from MLflow.")
            else:
                logger.warning(f"ONNX artifact not found in MLflow run {run_id}. Attempting disk fallback...")
        else:
            logger.warning(f"No registered model version found in MLflow registry for '{model_name}'. Attempting disk fallback...")

    except Exception as mlflow_err:
        logger.warning(f"Could not load model from MLflow Registry ({mlflow_err}). Attempting disk fallback...")

    # 3. Disk fallback if MLflow loading did not succeed
    if not loaded_from_mlflow:
        disk_onnx = os.getenv("ONNX_MODEL_PATH", "./models/lightgbm_model.onnx")
        default_fe = "./models/encoders/feature_engineer.pkl" if os.path.exists("./models/encoders/feature_engineer.pkl") else "./models/feature_engineer.pkl"
        disk_fe = os.getenv("FEATURE_ENGINEER_PATH", default_fe)

        if os.path.exists(disk_onnx):
            try:
                import onnxruntime as rt
                onnx_session = rt.InferenceSession(disk_onnx, providers=["CPUExecutionProvider"])
                model_version = "disk-local"
                logger.info(f"Loaded ONNX model from disk fallback: {disk_onnx}")
            except Exception as disk_err:
                logger.error(f"Failed to load ONNX model from disk: {disk_err}")

        if os.path.exists(disk_fe):
            try:
                from ml.features.temporal import TemporalFeatureEngineer
                feature_engineer = TemporalFeatureEngineer().load(disk_fe)
                logger.info(f"Loaded feature engineer from disk fallback: {disk_fe}")
            except Exception as fe_err:
                logger.error(f"Failed to load feature engineer from disk: {fe_err}")

    app.state.onnx_session = onnx_session
    app.state.feature_engineer = feature_engineer
    app.state.model_version = model_version

    # Set Prometheus info metric
    MODEL_VERSION_INFO.info({
        "version": model_version,
        "source": "mlflow" if loaded_from_mlflow else ("disk" if onnx_session else "fallback"),
    })
    DRIFT_ALERT_GAUGE.set(0)

    logger.info(f"Serving layer startup complete. Active model version: {model_version}")

    yield

    logger.info("Shutting down TaxiTrack Serving Layer...")


def create_app() -> FastAPI:
    """Factory creating and configuring the FastAPI application instance."""
    application = FastAPI(
        title="TaxiTrack Serving API",
        description="High-performance analytics, demand forecasting, and spatial network intelligence for NYC Taxi operations.",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS configuration
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API routers
    application.include_router(analytics_router)
    application.include_router(forecast_router)
    application.include_router(network_router)

    # Health check endpoint
    @application.get("/health", tags=["health"])
    def health_check():
        return {
            "status": "healthy",
            "model_version": getattr(application.state, "model_version", "unknown"),
            "onnx_loaded": getattr(application.state, "onnx_session", None) is not None,
        }

    # Root index
    @application.get("/", tags=["root"])
    def root():
        return {
            "service": "TaxiTrack Serving API",
            "version": "1.0.0",
            "docs_url": "/docs",
            "metrics_url": "/metrics",
            "health_url": "/health",
        }

    # Expose Prometheus metrics endpoint
    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
    )
    instrumentator.instrument(application).expose(application, endpoint="/metrics")

    return application


app = create_app()
