# airflow/dags/ml_drift_monitoring_dag.py
"""
Airflow MLOps DAG: Scheduled Drift Monitoring & Automated Retraining Trigger
Orchestrates:
  1. Multi-Type Drift Detection via Evidently AI (Feature, Target, Quality Drift)
  2. Gated ShortCircuit verification of drift severity thresholds
  3. Automated triggering of the Model Retraining DAG via TriggerDagRunOperator
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from core.logging import get_logger

logger = get_logger("taxitrack_drift_dag")


def check_evidently_drift_callable(**kwargs) -> dict:
    """
    Execute Evidently AI drift analysis comparing recent operational data vs baseline.
    Pushes structured drift verdict to Airflow XCom.
    """
    ds_str = kwargs.get("ds", datetime.now().strftime("%Y-%m-%d"))
    logger.info(f"Triggering Evidently multi-type drift evaluation at cutoff: {ds_str}")

    from ml.monitoring import DriftMonitoringPipeline

    pipeline = DriftMonitoringPipeline()
    verdict = pipeline.run(end_date=ds_str)

    logger.info(
        f"Drift evaluation completed: Severity={verdict.get('drift_severity')} | "
        f"Alert Trigger={verdict.get('alert_trigger')} | Action={verdict.get('recommended_action')}"
    )
    return verdict


def evaluate_drift_gate_callable(**kwargs) -> bool:
    """
    Inspect drift evaluation from upstream task.
    Returns True to proceed to TriggerDagRunOperator if drift is detected,
    or False to cleanly skip downstream tasks if data distributions are healthy.
    """
    ti = kwargs["ti"]
    verdict = ti.xcom_pull(task_ids="evaluate_evidently_drift")

    if not verdict:
        logger.warning("No drift verdict received from upstream task. Skipping retrain trigger.")
        return False

    alert_trigger = verdict.get("alert_trigger", False)
    severity = verdict.get("drift_severity", "HEALTHY")

    if alert_trigger:
        logger.info(
            f"🚨 DRIFT GATE BREACHED (Severity: {severity})! "
            f"Proceeding to trigger model retraining pipeline."
        )
        return True
    else:
        logger.info(
            f"✅ DRIFT GATE HEALTHY (Severity: {severity}). "
            f"No distribution shift detected. Skipping retraining trigger."
        )
        return False


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="ml_drift_monitoring_dag",
    default_args=default_args,
    description="Periodically monitors NYC Taxi feature/target drift via Evidently AI and triggers retrain on breach",
    schedule_interval="@weekly",
    start_date=datetime(2020, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "monitoring", "evidently", "drift", "mlops"],
) as dag:

    evaluate_drift_task = PythonOperator(
        task_id="evaluate_evidently_drift",
        python_callable=check_evidently_drift_callable,
        provide_context=True,
    )

    drift_gate_task = ShortCircuitOperator(
        task_id="drift_threshold_gate",
        python_callable=evaluate_drift_gate_callable,
        provide_context=True,
    )

    trigger_retrain_task = TriggerDagRunOperator(
        task_id="trigger_model_retrain_dag",
        trigger_dag_id="ml_model_retrain_and_evaluate_dag",
        reset_dag_run=True,
        wait_for_completion=False,
    )

    evaluate_drift_task >> drift_gate_task >> trigger_retrain_task
