# airflow/dags/ml_retrain_dag.py
"""
Airflow MLOps DAG: Model Retraining & Spatial Network Analytics
Orchestrates:
  1. Automated Model Retraining, Metric Evaluation & Model Registry Updates via MLflow
  2. Spatial Network Centrality & Flow Analysis via NetworkX

Triggered on-demand, scheduled, or triggered by ml_drift_monitoring_dag upon drift alert.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from core.logging import get_logger

logger = get_logger("taxitrack_retrain_dag")


def retrain_model_callable(**kwargs) -> dict:
    """Execute end-to-end ML training pipeline with MLflow logging and ONNX export."""
    ds_str = kwargs.get("ds", datetime.now().strftime("%Y-%m-%d"))
    dag_run = kwargs.get("dag_run")
    conf = dag_run.conf if dag_run and dag_run.conf else {}

    end_date = conf.get("end_date", ds_str)
    start_date = conf.get("start_date")
    model_type = conf.get("model_type")

    logger.info(f"Triggering ML training pipeline up to date cutoff: {end_date} (model: {model_type or 'default'})")

    from ml.pipeline import MLTrainingPipeline
    pipeline = MLTrainingPipeline()
    result = pipeline.run(end_date=end_date, start_date=start_date, model_type=model_type)

    logger.info(f"Model retraining completed successfully: {result}")
    return result


def analyze_spatial_network_callable(**kwargs) -> dict:
    """Execute spatial network centrality and corridor flow analysis."""
    ds_str = kwargs.get("ds", datetime.now().strftime("%Y-%m-%d"))
    dag_run = kwargs.get("dag_run")
    conf = dag_run.conf if dag_run and dag_run.conf else {}

    pickup_month = conf.get("pickup_month", ds_str[:7] if len(ds_str) >= 7 else None)
    logger.info(f"Triggering spatial transit network analysis for month: {pickup_month}")

    from ml.graph import SpatialNetworkAnalyzer
    analyzer = SpatialNetworkAnalyzer()
    summary = analyzer.run(pickup_month=pickup_month, output_dir="artifacts/network_analytics")

    logger.info(f"Spatial network analysis completed. Computed metrics for {len(summary)} zones.")
    return analyzer.metrics


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="ml_model_retrain_and_evaluate_dag",
    default_args=default_args,
    description="Orchestrates model retraining, MLflow evaluation, ONNX export, and spatial network analytics",
    schedule_interval=None,  # Event-driven: triggered by drift monitoring DAG or manual run
    start_date=datetime(2020, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "lightgbm", "mlflow", "onnx", "networkx", "retrain"],
) as dag:

    retrain_task = PythonOperator(
        task_id="retrain_demand_model",
        python_callable=retrain_model_callable,
        provide_context=True,
    )

    spatial_analysis_task = PythonOperator(
        task_id="analyze_spatial_network",
        python_callable=analyze_spatial_network_callable,
        provide_context=True,
    )

    retrain_task >> spatial_analysis_task

