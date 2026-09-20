# src/ml/monitoring/pipeline.py
"""
Drift Monitoring Pipeline (Single Responsibility Principle)
Decoupled pipeline that periodically extracts reference baseline vs recent operational
windows from ClickHouse, executes Evidently AI drift detection, and logs metrics.
"""
from typing import Dict, Any, Optional
from datetime import datetime
from core.ml_config import ml_config, MLConfig
from core.logging import get_logger
from ml.data.base import BaseDemandRepository
from ml.data.clickhouse import ClickHouseFeatureRepository
from ml.features.temporal import TemporalFeatureEngineer
from ml.monitoring.detector import EvidentlyDriftDetector

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

    def run(
        self,
        end_date: Optional[str] = None,
        test_months: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute drift monitoring comparing historical baseline against recent operational data.

        Parameters
        ----------
        end_date : str, optional
            Evaluation cutoff date ('YYYY-MM-DD'). Defaults to config or today.
        test_months : int, optional
            Recent operational lookback window in months. Defaults to config test_months.

        Returns
        -------
        Dict[str, Any]
            Evidently drift verdict dictionary with alert trigger.
        """
        cfg_data = self.config.data
        cfg_feat = self.config.features
        cutoff = end_date or datetime.now().strftime("%Y-%m-%d")
        t_months = test_months if test_months is not None else cfg_data.test_months

        logger.info(
            f"=== [START DRIFT MONITORING] Window: {cfg_data.start_date} to {cutoff} "
            f"(Operational Holdout: {t_months} months) ==="
        )

        # 1. Fetch data from ClickHouse
        raw_df = self.repo.get_demand_data(start_date=cfg_data.start_date, end_date=cutoff)
        if raw_df.empty:
            logger.warning(f"No operational data found up to {cutoff} for drift monitoring.")
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

        logger.info(
            f"=== [DRIFT MONITORING COMPLETED] Severity: {verdict.get('drift_severity')} | "
            f"Alert Trigger: {verdict.get('alert_trigger')} ==="
        )
        return verdict
