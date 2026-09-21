# airflow/dags/ml_retrain_dag.py
"""
Airflow MLOps DAG: Model Retraining & Spatial Network Analytics
Orchestrates:
  1. Automated Model Retraining, Metric Evaluation & Model Registry Updates via MLflow
  2. Spatial Network Centrality & Flow Analysis via NetworkX

Triggered on-demand, scheduled, or triggered by ml_drift_monitoring_dag upon drift alert.

Required dag_run.conf keys:
  - start_date (str): Training window start, e.g. '2019-01-01'
  - end_date   (str): Training window end,   e.g. '2020-07-01'
  - model_type (str, optional): Override model architecture, e.g. 'XGBoost'
"""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from airflow import DAG
from airflow.operators.python import PythonOperator
from core.logging import get_logger

logger = get_logger("taxitrack_retrain_dag")
cairo_tz = ZoneInfo("Africa/Cairo")


def retrain_model_callable(**kwargs) -> dict:
    """Execute end-to-end ML training pipeline with MLflow logging and ONNX export."""
    cairo_now = datetime.now(cairo_tz)
    cairo_now_str = cairo_now.strftime("%Y-%m-%d %H:%M:%S")
    dag_run = kwargs.get("dag_run")
    conf = dag_run.conf if dag_run and dag_run.conf else {}

    start_date = conf.get("start_date")
    end_date = conf.get("end_date")
    model_type = conf.get("model_type")

    if not start_date or not end_date:
        raise ValueError(
            "ml_model_retrain_and_evaluate_dag requires 'start_date' and 'end_date' "
            "in dag_run.conf. Trigger with: "
            "{'start_date': 'YYYY-MM-DD', 'end_date': 'YYYY-MM-DD'}"
        )

    logger.info(
        f"Triggering ML training pipeline: {start_date} → {end_date} "
        f"(model: {model_type or 'default'}) | Run Time (Cairo): {cairo_now_str}"
    )

    from ml.pipeline import MLTrainingPipeline
    pipeline = MLTrainingPipeline()
    result = pipeline.run(start_date=start_date, end_date=end_date, model_type=model_type)

    result["retrained_at_cairo"] = cairo_now_str
    result["timezone"] = "Africa/Cairo"

    logger.info(f"Model retraining completed successfully: {result}")
    return result


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
    description="Orchestrates model retraining, MLflow evaluation, and ONNX export",
    schedule_interval=None,  # Event-driven: triggered by drift monitoring DAG or manual run
    start_date=datetime(2020, 1, 1, tzinfo=cairo_tz),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "lightgbm", "mlflow", "onnx", "retrain"],
) as dag:

    retrain_task = PythonOperator(
        task_id="retrain_demand_model",
        python_callable=retrain_model_callable,
        provide_context=True,
    )
