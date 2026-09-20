# src/ml/monitoring/detector.py
"""
Evidently AI Multi-Type Drift Detection & Observability Engine
Monitors and quantifies distribution shift across three key dimensions using native Evidently AI:
  1. Feature / Covariate Drift (DataDriftPreset)
  2. Target / Demand Drift (TargetDriftPreset)
  3. Data Quality & Schema Drift (DataQualityPreset)
"""
import os
import json
import time
from typing import Dict, Any, List, Optional
import pandas as pd
from core.logging import get_logger

logger = get_logger(__name__)


class EvidentlyDriftDetector:
    """
    Comprehensive drift evaluation suite monitoring feature distributions,
    target demand dynamics, and data quality using native Evidently AI.
    """

    def __init__(
        self,
        drift_share_threshold: float = 0.3,
        target_drift_threshold: float = 0.05,
        confidence: float = 0.95,
    ):
        self.drift_share_threshold = drift_share_threshold
        self.target_drift_threshold = target_drift_threshold
        self.confidence = confidence

    def analyze_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        target_col: str = "total_trips",
        feature_cols: Optional[List[str]] = None,
        output_dir: str = "./reports/drift",
    ) -> Dict[str, Any]:
        """
        Execute multi-type drift analysis comparing Reference vs Current datasets.

        Parameters
        ----------
        reference_data : pd.DataFrame
            Baseline historical training dataset.
        current_data : pd.DataFrame
            Newly ingested operational window.
        target_col : str
            Target column name ('total_trips').
        feature_cols : list of str, optional
            List of feature names to monitor.
        output_dir : str
            Directory to export HTML report and JSON summary.

        Returns
        -------
        Dict[str, Any]
            Structured drift metrics, test verdicts, and actionable MLOps recommendation.
        """
        from evidently.legacy.report import Report
        from evidently.legacy.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset
        from evidently.legacy.pipeline.column_mapping import ColumnMapping

        os.makedirs(output_dir, exist_ok=True)
        start_time = time.perf_counter()

        # Isolate features to analyze
        if feature_cols is None:
            exclude = ["pickup_datetime", "pickup_date", target_col]
            feature_cols = [c for c in reference_data.columns if c in current_data.columns and c not in exclude]

        # Classify numerical vs categorical features
        num_features = []
        cat_features = []
        for c in feature_cols:
            if str(reference_data[c].dtype) in ("category", "object") or not pd.api.types.is_numeric_dtype(reference_data[c]):
                cat_features.append(c)
            else:
                num_features.append(c)

        logger.info(
            f"Initiating Evidently multi-type drift analysis across {len(feature_cols)} features "
            f"({len(num_features)} numeric, {len(cat_features)} categorical): "
            f"Reference={len(reference_data):,} rows, Current={len(current_data):,} rows."
        )

        # Prepare clean working copies
        selected_cols = [c for c in feature_cols + [target_col] if c in reference_data.columns and c in current_data.columns]
        ref_df = reference_data[selected_cols].copy()
        curr_df = current_data[selected_cols].copy()

        # Subsample if dataset is exceedingly large for fast, responsive HTML rendering
        max_rows = 50_000
        if len(ref_df) > max_rows:
            ref_df = ref_df.sample(n=max_rows, random_state=42)
        if len(curr_df) > max_rows:
            curr_df = curr_df.sample(n=max_rows, random_state=42)

        column_mapping = ColumnMapping(
            target=target_col if target_col in selected_cols else None,
            numerical_features=num_features,
            categorical_features=cat_features,
        )

        # 1. Run native Evidently AI Report with presets
        html_report_path = os.path.join(output_dir, "evidently_drift_report.html")
        json_report_path = os.path.join(output_dir, "drift_summary.json")

        report = Report(metrics=[
            DataDriftPreset(drift_share=self.drift_share_threshold),
            TargetDriftPreset(),
            DataQualityPreset(),
        ])

        report.run(reference_data=ref_df, current_data=curr_df, column_mapping=column_mapping)

        # Save native interactive HTML visualization & raw JSON
        report.save_html(html_report_path)
        logger.info(f"Evidently interactive HTML dashboard exported to: {html_report_path}")

        raw_dict = report.as_dict()

        # 2. Extract structured metrics from Evidently report
        metrics_list = raw_dict.get("metrics", [])

        # Feature / Dataset Drift
        drift_table_metric = next((m.get("result", {}) for m in metrics_list if m.get("metric") == "DataDriftTable"), {})
        dataset_drift = drift_table_metric.get("dataset_drift", False)
        number_of_drifted = drift_table_metric.get("number_of_drifted_columns", 0)
        share_of_drifted = round(float(drift_table_metric.get("share_of_drifted_columns", 0.0)) * 100.0, 2)
        drift_by_cols_raw = drift_table_metric.get("drift_by_columns", {})

        drifted_features = [col for col, info in drift_by_cols_raw.items() if info.get("drift_detected", False)]
        feature_drift_summary = {}
        for col, info in drift_by_cols_raw.items():
            feature_drift_summary[col] = {
                "type": info.get("column_type"),
                "drift_detected": info.get("drift_detected", False),
                "drift_score": round(float(info.get("drift_score", 0.0)), 6),
                "stattest_name": info.get("stattest_name"),
                "stattest_threshold": info.get("stattest_threshold"),
            }

        # Target Drift
        col_drift_metrics = [m.get("result", {}) for m in metrics_list if m.get("metric") == "ColumnDriftMetric"]
        target_info = next((m for m in col_drift_metrics if m.get("column_name") == target_col), {})
        target_drift_detected = target_info.get("drift_detected", False)
        target_drift_score = round(float(target_info.get("drift_score", 1.0)), 6) if target_info else 1.0
        target_summary = {
            "target_name": target_col,
            "drift_detected": target_drift_detected,
            "drift_score": target_drift_score,
            "stattest_name": target_info.get("stattest_name", "K-S"),
            "threshold": target_info.get("stattest_threshold", self.target_drift_threshold),
        }

        # Data Quality Issues
        quality_issues = []
        for m in metrics_list:
            if m.get("metric") == "DatasetMissingValuesMetric":
                res = m.get("result", {})
                curr_missing = res.get("current", {}).get("share_of_missing_values", 0.0)
                ref_missing = res.get("reference", {}).get("share_of_missing_values", 0.0)
                if curr_missing > (ref_missing + 0.05):
                    quality_issues.append(
                        f"Dataset missing values spiked from {ref_missing * 100:.1f}% to {curr_missing * 100:.1f}%"
                    )

        # 3. Overall MLOps Severity Verdict
        if dataset_drift and target_drift_detected:
            severity = "CRITICAL"
            recommended_action = "RETRAIN_MODEL_IMMEDIATELY"
        elif dataset_drift or target_drift_detected:
            severity = "MODERATE"
            recommended_action = "TRIGGER_RETRAINING_PIPELINE"
        elif share_of_drifted > 15.0 or len(quality_issues) > 0:
            severity = "LOW"
            recommended_action = "MONITOR_CLOSELY"
        else:
            severity = "HEALTHY"
            recommended_action = "NO_ACTION"

        duration = round(time.perf_counter() - start_time, 3)

        verdict = {
            "dataset_drift": dataset_drift,
            "drift_severity": severity,
            "recommended_action": recommended_action,
            "alert_trigger": bool(dataset_drift or target_drift_detected),
            "number_of_features_monitored": len(feature_cols),
            "number_of_drifted_features": len(drifted_features),
            "drifted_feature_share_pct": share_of_drifted,
            "drift_threshold_pct": round(self.drift_share_threshold * 100.0, 2),
            "drifted_features": drifted_features,
            "target_drift": target_summary,
            "data_quality_issues": quality_issues,
            "drift_by_columns": feature_drift_summary,
            "computation_duration_seconds": duration,
            "html_report_path": html_report_path,
            "json_report_path": json_report_path,
        }

        with open(json_report_path, "w", encoding="utf-8") as f:
            json.dump(verdict, f, indent=2)

        logger.info(
            f"\n====================================================================\n"
            f"  EVIDENTLY MULTI-TYPE DRIFT VERDICT\n"
            f"  DATASET DRIFT DETECTED: {'🚨 YES' if dataset_drift else '✅ NO'}\n"
            f"  DRIFTED FEATURE SHARE:  {share_of_drifted}%\n"
            f"  TARGET DEMAND DRIFT:    {'🚨 YES' if target_drift_detected else '✅ NO'}\n"
            f"  OVERALL DRIFT SEVERITY: {severity}\n"
            f"  RECOMMENDED ACTION:     {recommended_action}\n"
            f"  REPORT EXPORTED:        {html_report_path}\n"
            f"===================================================================="
        )
        return verdict
