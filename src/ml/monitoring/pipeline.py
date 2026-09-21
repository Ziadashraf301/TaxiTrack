# src/ml/monitoring/pipeline.py
"""
Drift Monitoring Pipeline (Single Responsibility Principle)
Decoupled pipeline that periodically extracts reference baseline vs recent operational
windows from ClickHouse, executes Evidently AI drift detection, and logs metrics.
"""
import os
from typing import Dict, Any, Optional
from datetime import datetime
from core.ml_config import ml_config, MLConfig
from core.logging import get_logger
from ml.data.base import BaseDemandRepository
from ml.data.clickhouse import ClickHouseFeatureRepository
from ml.features.temporal import TemporalFeatureEngineer
from ml.monitoring.detector import EvidentlyDriftDetector
from ml.tracking.base import BaseExperimentTracker
from ml.tracking.mlflow import MLflowExperimentTracker

logger = get_logger(__name__)


class DriftMonitoringPipeline:
    """
    Independent orchestrator for continuous data observability and distribution shift detection.
    """

    def __init__(
        self,
        config: Optional[MLConfig] = None,
        feature_repo: Optional[BaseDemandRepository] = None,
        detector: Optional[EvidentlyDriftDetector] = None,
        tracker: Optional[BaseExperimentTracker] = None,
        reports_dir: str = "./reports/drift",
    ):
        self.config = config or ml_config
        self.repo = feature_repo or ClickHouseFeatureRepository()
        self.detector = detector or EvidentlyDriftDetector(
            drift_share_threshold=self.config.monitoring.evidently.drift_share_threshold,
            target_drift_threshold=self.config.monitoring.evidently.target_drift_threshold,
            confidence=self.config.monitoring.evidently.confidence_level,
        )
        self.reports_dir = reports_dir

        if tracker is not None:
            self.tracker = tracker
        else:
            try:
                self.tracker = MLflowExperimentTracker(
                    tracking_uri=self.config.mlflow.get_tracking_uri(),
                    experiment_name="taxitrack_drift_monitoring",
                )
            except Exception as e:
                logger.warning(f"Could not initialize MLflow tracking for drift monitoring: {e}. Tracking disabled.")
                self.tracker = None

    def run(
        self,
        start_date: str,
        end_date: str,
    ) -> Dict[str, Any]:
        """
        Execute drift monitoring comparing historical baseline against recent operational data.

        Parameters
        ----------
        start_date : str
            Fetch window start ('YYYY-MM-DD'). Must be provided by Airflow dag_run.conf.
        end_date : str
            Evaluation cutoff date ('YYYY-MM-DD'). Must be provided by Airflow dag_run.conf.

        The train/validation split uses cfg_data.test_months (fixed pipeline parameter from config)
        to separate reference baseline from current operational window.

        Returns
        -------
        Dict[str, Any]
            Evidently drift verdict dictionary with alert trigger.
        """
        if not start_date or not end_date:
            raise ValueError(
                "DriftMonitoringPipeline.run() requires both 'start_date' and 'end_date'. "
                "These must be provided by Airflow dag_run.conf, not hardcoded in config."
            )

        cfg_data = self.config.data
        cfg_feat = self.config.features
        t_months = cfg_data.test_months  # Fixed pipeline design parameter from config

        logger.info(
            f"=== [START DRIFT MONITORING] Window: {start_date} to {end_date} "
            f"(Operational Holdout: {t_months} months) ==="
        )

        # 1. Fetch data from ClickHouse using Airflow-provided window
        raw_df = self.repo.get_demand_data(start_date=start_date, end_date=end_date)
        if raw_df.empty:
            logger.warning(f"No operational data found for window [{start_date}, {end_date}] for drift monitoring.")
            return {
                "dataset_drift": False,
                "drift_severity": "UNKNOWN",
                "recommended_action": "NO_ACTION",
                "alert_trigger": False,
                "reason": "empty_dataset",
            }

        # 2. Extract baseline (reference) vs operational (current) features via Lookback Buffer
        fe = TemporalFeatureEngineer(
            lag_hours=cfg_feat.lag_hours,
            rolling_windows=cfg_feat.rolling_windows,
            group_col=cfg_data.group_col,
            target_col=cfg_data.target_col,
            timestamp_col=cfg_data.timestamp_col,
        )
        X_ref, X_curr, y_ref, y_curr = fe.prepare_train_val_split(raw_df, test_months=t_months)

        ref_combined = X_ref.copy()
        ref_combined[cfg_data.target_col] = y_ref
        curr_combined = X_curr.copy()
        curr_combined[cfg_data.target_col] = y_curr

        # 3. Analyze drift via Evidently AI
        verdict = self.detector.analyze_drift(
            reference_data=ref_combined,
            current_data=curr_combined,
            target_col=cfg_data.target_col,
            feature_cols=fe.feature_columns,
            output_dir=self.reports_dir,
        )

        # 4. Log drift observability metrics & dashboard artifacts to MLflow
        if self.tracker:
            try:
                from zoneinfo import ZoneInfo
                cairo_now_str = datetime.now(ZoneInfo("Africa/Cairo")).strftime("%Y-%m-%d %H:%M:%S")

                run_name = f"drift_eval_{end_date}"
                target_info = verdict.get("target_drift", {})
                run_tags = {
                    "pipeline": "evidently_drift_monitoring",
                    "start_date": str(start_date),
                    "end_date": str(end_date),
                    "drift_severity": str(verdict.get("drift_severity", "UNKNOWN")),
                    "dataset_drift": str(verdict.get("dataset_drift", False)),
                    "alert_trigger": str(verdict.get("alert_trigger", False)),
                    "recommended_action": str(verdict.get("recommended_action", "NO_ACTION")),
                    "evaluated_at_cairo": cairo_now_str,
                    "timezone": "Africa/Cairo",
                }
                with self.tracker.start_run(run_name=run_name, tags=run_tags) as run:
                    run_id = getattr(run.info, "run_id", "unknown")
                    logger.info(f"Logging drift evaluation to MLflow (run_id: {run_id})...")

                    # 4.1 Parameters
                    self.tracker.log_params({
                        "start_date": start_date,
                        "end_date": end_date,
                        "test_months": t_months,
                        "target_col": cfg_data.target_col,
                        "num_features_monitored": verdict.get("number_of_features_monitored", 0),
                        "drift_threshold_pct": verdict.get("drift_threshold_pct", 30.0),
                        "drift_severity": verdict.get("drift_severity", "UNKNOWN"),
                        "recommended_action": verdict.get("recommended_action", "NO_ACTION"),
                        "evaluated_at_cairo": cairo_now_str,
                        "timezone": "Africa/Cairo",
                    })

                    # 4.2 Metrics
                    metrics_to_log = {
                        "drifted_feature_share_pct": float(verdict.get("drifted_feature_share_pct", 0.0)),
                        "number_of_drifted_features": float(verdict.get("number_of_drifted_features", 0)),
                        "computation_duration_seconds": float(verdict.get("computation_duration_seconds", 0.0)),
                        "target_drift_score": float(target_info.get("drift_score", 0.0)),
                        "target_drift_detected": 1.0 if target_info.get("drift_detected", False) else 0.0,
                        "dataset_drift_detected": 1.0 if verdict.get("dataset_drift", False) else 0.0,
                    }
                    for col, details in verdict.get("drift_by_columns", {}).items():
                        if isinstance(details, dict) and "drift_score" in details:
                            metrics_to_log[f"drift_score_{col}"] = float(details["drift_score"])

                    self.tracker.log_metrics(metrics_to_log)

                    # 4.3 Report Artifacts (HTML Dashboard & JSON Summary)
                    html_path = verdict.get("html_report_path")
                    json_path = verdict.get("json_report_path")
                    if html_path and os.path.exists(html_path):
                        self.tracker.log_artifact(html_path, artifact_path="reports")
                    if json_path and os.path.exists(json_path):
                        self.tracker.log_artifact(json_path, artifact_path="reports")

                    logger.info(f"Successfully logged Evidently drift artifacts to MLflow experiment '{self.tracker.experiment_name}'.")
            except Exception as mlflow_err:
                logger.warning(f"MLflow drift experiment logging failed: {mlflow_err}")

        logger.info(
            f"=== [DRIFT MONITORING COMPLETED] Severity: {verdict.get('drift_severity')} | "
            f"Alert Trigger: {verdict.get('alert_trigger')} ==="
        )
        return verdict
