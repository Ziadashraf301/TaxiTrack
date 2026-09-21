# tests/test_drift_monitoring.py
"""
Unit and integration tests for Evidently AI drift monitoring suite.
Verifies multi-type drift detection (feature drift, target drift, data quality),
HTML dashboard generation, JSON summary export, and alert triggering.
"""
import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from ml.monitoring.detector import EvidentlyDriftDetector
from ml.monitoring.pipeline import DriftMonitoringPipeline
from ml.data.base import BaseDemandRepository


class MockDemandRepository(BaseDemandRepository):
    """In-memory mock repository for isolated unit testing."""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def get_demand_data(self, start_date=None, end_date=None, service_type=None) -> pd.DataFrame:
        return self._df.copy()

    def get_demand_summary(self):
        if self._df.empty:
            return None, None, 0
        min_d = str(self._df["pickup_datetime"].min())
        max_d = str(self._df["pickup_datetime"].max())
        return min_d, max_d, len(self._df)


@pytest.fixture
def synthetic_datasets():
    """Generates synthetic reference and drifted datasets for NYC taxi demand."""
    np.random.seed(42)
    n_samples = 400

    # Reference data: Normal baseline
    ref_df = pd.DataFrame({
        "location_id": np.random.choice([100, 101, 102], size=n_samples),
        "hour": np.random.randint(0, 24, size=n_samples),
        "day_of_week": np.random.randint(0, 7, size=n_samples),
        "lag_1h": np.random.normal(loc=50.0, scale=10.0, size=n_samples),
        "lag_24h": np.random.normal(loc=50.0, scale=10.0, size=n_samples),
        "rolling_mean_3h": np.random.normal(loc=50.0, scale=8.0, size=n_samples),
        "total_trips": np.random.normal(loc=50.0, scale=10.0, size=n_samples),
    })

    # Current data: Identical distribution (Healthy / No drift)
    healthy_curr_df = pd.DataFrame({
        "location_id": np.random.choice([100, 101, 102], size=n_samples),
        "hour": np.random.randint(0, 24, size=n_samples),
        "day_of_week": np.random.randint(0, 7, size=n_samples),
        "lag_1h": np.random.normal(loc=50.0, scale=10.0, size=n_samples),
        "lag_24h": np.random.normal(loc=50.0, scale=10.0, size=n_samples),
        "rolling_mean_3h": np.random.normal(loc=50.0, scale=8.0, size=n_samples),
        "total_trips": np.random.normal(loc=50.0, scale=10.0, size=n_samples),
    })

    # Current data: Shifted distribution (Drifted)
    drifted_curr_df = pd.DataFrame({
        "location_id": np.random.choice([100, 101, 102], size=n_samples),
        "hour": np.random.randint(0, 24, size=n_samples),
        "day_of_week": np.random.randint(0, 7, size=n_samples),
        "lag_1h": np.random.normal(loc=180.0, scale=25.0, size=n_samples),
        "lag_24h": np.random.normal(loc=200.0, scale=30.0, size=n_samples),
        "rolling_mean_3h": np.random.normal(loc=190.0, scale=20.0, size=n_samples),
        "total_trips": np.random.normal(loc=220.0, scale=35.0, size=n_samples),
    })

    return ref_df, healthy_curr_df, drifted_curr_df


def test_evidently_drift_detector_healthy(synthetic_datasets):
    """Verify that identical distributions result in HEALTHY status and no alerts."""
    ref_df, healthy_curr_df, _ = synthetic_datasets

    with tempfile.TemporaryDirectory() as tmpdir:
        detector = EvidentlyDriftDetector(drift_share_threshold=0.3, target_drift_threshold=0.05)
        verdict = detector.analyze_drift(
            reference_data=ref_df,
            current_data=healthy_curr_df,
            target_col="total_trips",
            output_dir=tmpdir,
        )

        assert verdict["alert_trigger"] is False
        assert verdict["drift_severity"] in ("HEALTHY", "LOW")
        assert verdict["recommended_action"] in ("NO_ACTION", "MONITOR_CLOSELY")
        assert os.path.exists(verdict["html_report_path"])
        assert os.path.exists(verdict["json_report_path"])
        assert os.path.getsize(verdict["html_report_path"]) > 0


def test_evidently_drift_detector_drift_detected(synthetic_datasets):
    """Verify that heavily shifted distributions trigger drift alerts and retrain actions."""
    ref_df, _, drifted_curr_df = synthetic_datasets

    with tempfile.TemporaryDirectory() as tmpdir:
        detector = EvidentlyDriftDetector(drift_share_threshold=0.3, target_drift_threshold=0.05)
        verdict = detector.analyze_drift(
            reference_data=ref_df,
            current_data=drifted_curr_df,
            target_col="total_trips",
            output_dir=tmpdir,
        )

        assert verdict["alert_trigger"] is True
        assert verdict["drift_severity"] in ("MODERATE", "CRITICAL")
        assert verdict["recommended_action"] in ("TRIGGER_RETRAINING_PIPELINE", "RETRAIN_MODEL_IMMEDIATELY")
        assert verdict["number_of_drifted_features"] > 0
        assert verdict["drifted_feature_share_pct"] > 30.0


def test_drift_monitoring_pipeline_run():
    """Verify end-to-end execution of DriftMonitoringPipeline with mock repository."""
    # Create temporal dataset with 3 months of hourly data
    dates = pd.date_range("2024-01-01", "2024-03-31 23:00:00", freq="h")
    np.random.seed(123)
    df = pd.DataFrame({
        "pickup_datetime": dates,
        "pickup_hour": dates,
        "pickup_location_id": np.random.choice([100, 101], size=len(dates)),
        "total_trips": np.random.randint(10, 80, size=len(dates)),
        "avg_fare_amount": np.random.uniform(10.0, 40.0, size=len(dates)),
        "avg_trip_distance": np.random.uniform(1.0, 8.0, size=len(dates)),
    })

    mock_repo = MockDemandRepository(df)
    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = DriftMonitoringPipeline(feature_repo=mock_repo, reports_dir=tmpdir)
        verdict = pipeline.run(end_date="2024-03-31")

        assert "drift_severity" in verdict
        assert "alert_trigger" in verdict
        assert os.path.exists(verdict["html_report_path"])
