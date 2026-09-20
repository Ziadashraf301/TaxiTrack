# src/ml/tracking/mlflow.py
"""
MLflow Experiment Tracking & Model Registry Client (Adapter Pattern)
Manages experiment initialization, hyperparameter tracking, metric logging,
artifact persistence to MinIO S3, and model registry lifecycle transitions.
"""
import os
import tempfile
from typing import Dict, Any, Optional
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

from core.config import settings
from core.logging import get_logger
from ml.tracking.base import BaseExperimentTracker

logger = get_logger(__name__)


class MLflowExperimentTracker(BaseExperimentTracker):
    """
    Adapter interfacing with the MLflow Tracking Server and Model Registry.
    Guarantees environment configuration for S3 artifact storage in MinIO.
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ):
        self.tracking_uri = tracking_uri or settings.mlflow.tracking_uri
        self.experiment_name = experiment_name or settings.mlflow.experiment_name

        self._configure_environment()

        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        self.experiment_id = self._ensure_experiment()

        logger.info(
            f"MLflowTracker initialized (tracking_uri='{self.tracking_uri}', "
            f"experiment='{self.experiment_name}', id='{self.experiment_id}')."
        )

    def _configure_environment(self) -> None:
        """Inject MinIO credentials and endpoint for MLflow S3 artifact backend."""
        os.environ["MLFLOW_TRACKING_URI"] = self.tracking_uri
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = settings.mlflow.s3_endpoint_url
        os.environ["AWS_ACCESS_KEY_ID"] = settings.minio.root_user
        os.environ["AWS_SECRET_ACCESS_KEY"] = settings.minio.root_password

    def _ensure_experiment(self) -> str:
        """Create or retrieve existing MLflow experiment."""
        exp = mlflow.get_experiment_by_name(self.experiment_name)
        if exp is None:
            try:
                exp_id = mlflow.create_experiment(
                    name=self.experiment_name,
                    artifact_location=f"s3://{settings.mlflow.artifact_bucket}/{self.experiment_name}",
                )
                logger.info(f"Created new MLflow experiment '{self.experiment_name}' (ID: {exp_id}).")
                return exp_id
            except Exception as e:
                logger.warning(f"Could not create experiment with custom artifact location: {e}. Falling back to default.")
                return mlflow.create_experiment(name=self.experiment_name)
        return exp.experiment_id

    def start_run(self, run_name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager starting a tracked MLflow run."""
        self._configure_environment()
        run_tags = {"framework": "LightGBM", "dataset": "NYC_TLC_Taxi"}
        if tags:
            run_tags.update(tags)
        return mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=run_tags,
        )

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log training parameters and configuration."""
        clean_params = {}
        for k, v in params.items():
            if isinstance(v, (int, float, str, bool)):
                clean_params[k] = v
            else:
                clean_params[k] = str(v)[:250]
        mlflow.log_params(clean_params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log evaluation metrics."""
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Upload a file or directory to the MLflow artifact repository."""
        if os.path.exists(local_path):
            mlflow.log_artifact(local_path, artifact_path=artifact_path)
            logger.info(f"Uploaded artifact '{local_path}' to MLflow.")
        else:
            logger.warning(f"Artifact path '{local_path}' does not exist. Skipping.")

    def log_dataframe(self, df: pd.DataFrame, filename: str, artifact_path: Optional[str] = None) -> None:
        """Save and log a Pandas DataFrame as a CSV artifact."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_file = os.path.join(tmp_dir, filename)
            df.to_csv(tmp_file, index=False)
            mlflow.log_artifact(tmp_file, artifact_path=artifact_path)

    def log_lightgbm_model(
        self,
        lgb_model,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
    ):
        """Log LightGBM model and optionally register in the MLflow Model Registry."""
        try:
            from mlflow.lightgbm import log_model
            model_info = log_model(
                lgb_model=lgb_model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name,
            )
            logger.info(f"Successfully logged LightGBM model to MLflow (URI: {model_info.model_uri}).")
            return model_info
        except Exception as e:
            logger.error(f"Failed to log LightGBM model to MLflow: {e}", exc_info=True)
            raise

    def register_legacy_model(
        self,
        model_path: str,
        artifacts_dir: Optional[str] = None,
        model_name: str = "taxi-demand-forecaster",
        run_name: str = "baseline_2019_2025_pretrained",
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Ingest and register existing pre-trained model (.pkl) into MLflow as Version 1 baseline.
        """
        logger.info(f"Registering legacy baseline model from '{model_path}' into MLflow...")
        run_tags = {
            "source": "legacy_pre_trained",
            "coverage": "2019-01_to_2025-08",
            "model_type": "LightGBM",
            "status": "baseline",
        }
        if tags:
            run_tags.update(tags)

        with self.start_run(run_name=run_name, tags=run_tags):
            if metrics:
                self.log_metrics(metrics)

            self.log_artifact(model_path, artifact_path="model_artifacts")

            if artifacts_dir and os.path.exists(artifacts_dir):
                for item in os.listdir(artifacts_dir):
                    item_path = os.path.join(artifacts_dir, item)
                    if os.path.isfile(item_path):
                        self.log_artifact(item_path, artifact_path="model_artifacts")

            active_run = mlflow.active_run()
            run_id = active_run.info.run_id
            source_uri = f"runs:/{run_id}/model_artifacts"

            try:
                mv = self.client.create_model_version(
                    name=model_name,
                    source=source_uri,
                    run_id=run_id,
                    description="Pre-trained LightGBM baseline trained on 2019-2025 NYC taxi dataset.",
                    tags={"stage": "baseline_2025"},
                )
                logger.info(f"Registered model version: {mv.name} v{mv.version} (run_id: {run_id}).")
                return run_id
            except Exception as e:
                try:
                    self.client.create_registered_model(name=model_name, description="NYC Taxi Demand Forecasters")
                    mv = self.client.create_model_version(
                        name=model_name,
                        source=source_uri,
                        run_id=run_id,
                        description="Pre-trained LightGBM baseline trained on 2019-2025 NYC taxi dataset.",
                        tags={"stage": "baseline_2025"},
                    )
                    logger.info(f"Created registered model '{model_name}' and added v{mv.version}.")
                    return run_id
                except Exception as inner_e:
                    logger.warning(f"Could not register model in registry: {inner_e}. Run artifacts are safely logged.")
                    return run_id
