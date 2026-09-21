# src/ml/pipeline.py
"""
End-to-End ML Training Pipeline Orchestrator (Facade & Dependency Injection Patterns)
Coordinates feature extraction, temporal lookback buffer transformations,
model training via ForecasterFactory, MLflow experiment tracking, Model Registry updates,
and ONNX runtime artifact export.
"""
import os
import time
from typing import Dict, Any, Optional
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import joblib

from core.ml_config import ml_config, MLConfig
from ml.data.base import BaseDemandRepository
from ml.data.clickhouse import ClickHouseFeatureRepository
from ml.features.temporal import TemporalFeatureEngineer
from ml.models.factory import ForecasterFactory
from ml.serving.onnx import ONNXModelExporter
from ml.tracking.base import BaseExperimentTracker
from ml.tracking.mlflow import MLflowExperimentTracker
from ml.monitoring.detector import EvidentlyDriftDetector
from core.logging import get_logger

logger = get_logger(__name__)


class MLTrainingPipeline:
    """
    Facade orchestrating model training, evaluation, experiment tracking,
    and production ONNX artifact export.
    """

    def __init__(
        self,
        config: Optional[MLConfig] = None,
        feature_repo: Optional[BaseDemandRepository] = None,
        tracker: Optional[BaseExperimentTracker] = None,
        models_dir: str = "./models",
        reports_dir: str = "./reports/drift",
    ):
        self.config = config or ml_config
        self.repo = feature_repo or ClickHouseFeatureRepository()
        self.tracker = tracker or MLflowExperimentTracker(
            tracking_uri=self.config.mlflow.get_tracking_uri(),
            experiment_name=self.config.mlflow.get_experiment_name(),
        )
        self.models_dir = models_dir
        self.reports_dir = reports_dir

    def run(
        self,
        start_date: str,
        end_date: str,
        model_type: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        register_model: bool = True,
        check_drift: bool = False,
        export_onnx: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute full training workflow.

        Parameters
        ----------
        start_date : str
            Training window start date ('YYYY-MM-DD'). Must be provided by Airflow dag_run.conf.
        end_date : str
            Training window end date ('YYYY-MM-DD'). Must be provided by Airflow dag_run.conf.
        model_type : str, optional
            Override model architecture. Defaults to config model_type.
        hyperparameters : dict, optional
            Override model hyperparameters. Defaults to config hyperparameters.
        register_model : bool
            Whether to register the model in MLflow Model Registry.
        check_drift : bool
            Whether to run Evidently drift analysis on train vs validation split.
        export_onnx : bool
            Whether to export model to ONNX format and benchmark latency.
        """
        if not start_date or not end_date:
            raise ValueError(
                "MLTrainingPipeline.run() requires both 'start_date' and 'end_date'. "
                "These must be provided by Airflow dag_run.conf, not hardcoded in config."
            )

        cfg_data = self.config.data
        cfg_feat = self.config.features
        cfg_model = self.config.model
        cfg_mon = self.config.monitoring.evidently
        cfg_srv = self.config.serving

        m_type = model_type or cfg_model.model_type
        h_params = hyperparameters or cfg_model.hyperparameters

        logger.info(
            f"\n====================================================================\n"
            f"  TAXITRACK ML PIPELINE EXECUTION\n"
            f"  MODEL ARCHITECTURE: {m_type}\n"
            f"  TRAINING WINDOW:    {start_date} ➔ {end_date}\n"
            f"  VALIDATION MONTHS:  {cfg_data.test_months}\n"
            f"  TARGET MART:        {cfg_data.mart_table}\n"
            f"===================================================================="
        )
        total_start = time.perf_counter()

        # Step 1: Query features from repository
        df_raw = self.repo.get_demand_data(start_date=start_date, end_date=end_date)
        if df_raw.empty:
            raise ValueError(f"No training data found in ClickHouse for window [{start_date}, {end_date}].")

        # Step 2: Temporal Feature Engineering via Lookback Buffer Pattern (Zero Leakage)
        feature_engineer = TemporalFeatureEngineer(
            lag_hours=cfg_feat.lag_hours,
            rolling_windows=cfg_feat.rolling_windows,
            group_col=cfg_data.group_col,
            target_col=cfg_data.target_col,
            timestamp_col=cfg_data.timestamp_col,
        )

        logger.info(f"Executing Lookback Buffer split & feature engineering (validation holdout: {cfg_data.test_months} months)...")
        X_train, X_val, y_train, y_val = feature_engineer.prepare_train_val_split(
            df_raw, test_months=cfg_data.test_months
        )

        # Step 3: Model Instantiation via Factory & Training
        forecaster = ForecasterFactory.create(model_type=m_type, params=h_params)
        logger.info(f"Fitting {forecaster.name} model on {len(X_train):,} training records...")
        forecaster.fit(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            early_stopping_rounds=cfg_model.early_stopping_rounds,
        )

        # Step 4: Metric Evaluation
        logger.info("Evaluating model performance on validation holdout...")
        metrics = forecaster.evaluate(X_val, y_val)
        importances = forecaster.get_feature_importances()

        # Step 5: Optional Multi-Type Drift Analysis via Evidently AI
        drift_metrics = {}
        if check_drift:
            detector = EvidentlyDriftDetector(
                drift_share_threshold=cfg_mon.drift_share_threshold,
                target_drift_threshold=cfg_mon.target_drift_threshold,
                confidence=cfg_mon.confidence_level,
            )
            ref_combined = X_train.copy()
            ref_combined[cfg_data.target_col] = y_train
            val_combined = X_val.copy()
            val_combined[cfg_data.target_col] = y_val

            drift_metrics = detector.analyze_drift(
                reference_data=ref_combined,
                current_data=val_combined,
                target_col=cfg_data.target_col,
                feature_cols=feature_engineer.feature_columns,
                output_dir=self.reports_dir,
            )

        # Step 6: Persist Local Artifacts
        os.makedirs(self.models_dir, exist_ok=True)
        encoders_dir = os.path.join(self.models_dir, "encoders")
        os.makedirs(encoders_dir, exist_ok=True)

        model_path = os.path.join(self.models_dir, "lightgbm_model.pkl")
        fe_path = os.path.join(encoders_dir, "feature_engineer.pkl")
        mappings_path = os.path.join(encoders_dir, "category_mappings.pkl")

        forecaster.save(model_path)
        feature_engineer.save(fe_path)
        joblib.dump(feature_engineer.category_mappings, mappings_path)

        # Step 7: ONNX Export & Parity Validation (wired to ServingConfig)
        onnx_path = None
        onnx_benchmarks = {}
        if export_onnx and hasattr(forecaster, "model"):
            try:
                onnx_path = os.path.join(self.models_dir, "lightgbm_model.onnx")
                ONNXModelExporter.export(
                    forecaster.model,
                    feature_names=forecaster.feature_names,
                    output_path=onnx_path,
                    target_opset=cfg_srv.target_opset,
                )
                sample_test = X_val.head(100)
                is_valid, max_delta = ONNXModelExporter.validate_numerical_parity(
                    native_model=forecaster.model,
                    onnx_path=onnx_path,
                    sample_input=sample_test,
                    atol=cfg_srv.parity_tolerance,
                )
                if not is_valid:
                    logger.warning(
                        f"ONNX parity check FAILED. Max delta: {max_delta:.6e} exceeds "
                        f"tolerance {cfg_srv.parity_tolerance}. Serving predictions may diverge."
                    )
                onnx_benchmarks = ONNXModelExporter.benchmark_latency(
                    onnx_path=onnx_path,
                    sample_input=sample_test,
                    num_runs=cfg_srv.benchmark_runs,
                )
            except Exception as onnx_err:
                logger.warning(f"ONNX export/validation notice: {onnx_err}")

        # Step 8: MLflow Tracking & Model Registry Update
        run_name = f"{m_type.lower()}_train_{end_date}"
        target_drift_info = drift_metrics.get("target_drift", {})
        target_drift_flag = target_drift_info.get("drift_detected", False)

        cairo_now_str = datetime.now(ZoneInfo("Africa/Cairo")).strftime("%Y-%m-%d %H:%M:%S")

        run_tags = {
            "model_architecture": m_type,
            "trained_until": str(end_date),
            "start_date": str(start_date),
            "train_samples": str(len(X_train)),
            "val_samples": str(len(X_val)),
            "dataset_drift": str(drift_metrics.get("dataset_drift", False)),
            "target_drift": str(target_drift_flag),
            "drift_severity": str(drift_metrics.get("drift_severity", "HEALTHY")),
            "recommended_action": str(drift_metrics.get("recommended_action", "NO_ACTION")),
            "trained_at_cairo": cairo_now_str,
            "timezone": "Africa/Cairo",
        }

        with self.tracker.start_run(run_name=run_name, tags=run_tags) as run:
            run_id = run.info.run_id
            logger.info(f"Logging complete run provenance to MLflow (run_id: {run_id})...")

            # 1. Log all hyperparameters and pipeline configs
            self.tracker.log_params(h_params)
            self.tracker.log_params({
                "model_type": m_type,
                "start_date": start_date,
                "end_date": end_date,
                "test_months": cfg_data.test_months,
                "early_stopping_rounds": cfg_model.early_stopping_rounds,
                "target_table": cfg_data.mart_table,
                "num_features": len(feature_engineer.feature_columns),
                "lag_hours": str(cfg_feat.lag_hours),
                "rolling_windows": str(cfg_feat.rolling_windows),
                "drift_severity": drift_metrics.get("drift_severity", "HEALTHY"),
                "trained_at_cairo": cairo_now_str,
                "timezone": "Africa/Cairo",
            })

            # 2. Log evaluation and drift metrics
            self.tracker.log_metrics(metrics)
            if drift_metrics:
                self.tracker.log_metrics({
                    "drifted_feature_share_pct": drift_metrics.get("drifted_feature_share_pct", 0.0),
                    "number_of_drifted_features": float(drift_metrics.get("number_of_drifted_features", 0)),
                    "target_drift_score": float(target_drift_info.get("drift_score", 1.0)),
                })
            if onnx_benchmarks:
                self.tracker.log_metrics({f"onnx_{k}": v for k, v in onnx_benchmarks.items()})

            # 3. Log artifacts (Models, Mappings, Reports, Config)
            self.tracker.log_artifact(model_path, artifact_path="model")
            self.tracker.log_artifact(fe_path, artifact_path="feature_engineering")
            self.tracker.log_artifact(mappings_path, artifact_path="feature_engineering")
            if onnx_path and os.path.exists(onnx_path):
                self.tracker.log_artifact(onnx_path, artifact_path="onnx")
            self.tracker.log_dataframe(importances, "feature_importance.csv", artifact_path="reports")

            # Log SSOT YAML config
            cfg_file = "configs/ml_config.yaml"
            if os.path.exists(cfg_file):
                self.tracker.log_artifact(cfg_file, artifact_path="config")

            # Log Evidently drift HTML & JSON reports if executed
            html_rep = drift_metrics.get("html_report_path")
            json_rep = drift_metrics.get("json_report_path")
            if html_rep and os.path.exists(html_rep):
                self.tracker.log_artifact(html_rep, artifact_path="evidently_reports")
            if json_rep and os.path.exists(json_rep):
                self.tracker.log_artifact(json_rep, artifact_path="evidently_reports")

            # 4. Register in Model Registry
            if register_model and hasattr(forecaster, "model") and hasattr(self.tracker, "log_lightgbm_model"):
                try:
                    self.tracker.log_lightgbm_model(
                        forecaster.model,
                        artifact_path="lgb_model",
                        registered_model_name=self.config.mlflow.registered_model_name,
                    )
                except Exception as reg_err:
                    logger.warning(f"Model registry notice: {reg_err}")

        total_duration = time.perf_counter() - total_start
        logger.info(
            f"\n====================================================================\n"
            f"  PIPELINE EXECUTION COMPLETED IN {total_duration:.2f}s\n"
            f"  RUN ID:     {run_id}\n"
            f"  MAE:        {metrics['mae']}\n"
            f"  RMSE:       {metrics['rmse']}\n"
            f"  WAPE:       {metrics['wape_pct']}%\n"
            f"  R2 SCORE:   {metrics['r2_score']}\n"
            f"===================================================================="
        )

        return {
            "status": "success",
            "run_id": run_id,
            "metrics": metrics,
            "drift_metrics": drift_metrics,
            "duration_seconds": round(total_duration, 2),
            "model_path": model_path,
            "onnx_path": onnx_path,
            "onnx_benchmarks": onnx_benchmarks,
        }
