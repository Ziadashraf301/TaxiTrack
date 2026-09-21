# airflow/dags/ml_drift_monitoring_dag.py
"""
Airflow MLOps DAG: Scheduled Drift Monitoring & Automated Retraining Trigger
Orchestrates:
  1. Multi-Type Drift Detection via Evidently AI (Feature, Target, Quality Drift)
  2. Gated ShortCircuit verification of drift severity thresholds
  3. Automated triggering of the Model Retraining DAG via Airflow trigger API

Required dag_run.conf keys:
  - start_date (str): Data window start, e.g. '2019-01-01'
  - end_date   (str): Evaluation cutoff,  e.g. '2020-07-01'
"""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from core.logging import get_logger

logger = get_logger("taxitrack_drift_dag")
cairo_tz = ZoneInfo("Africa/Cairo")


def check_evidently_drift_callable(**kwargs) -> dict:
    """
    Execute Evidently AI drift analysis comparing recent operational data vs baseline.
    Pushes structured drift verdict to Airflow XCom.
    """
    cairo_now = datetime.now(cairo_tz)
    cairo_now_str = cairo_now.strftime("%Y-%m-%d %H:%M:%S")
    dag_run = kwargs.get("dag_run")
    conf = dag_run.conf if dag_run and dag_run.conf else {}

    start_date = conf.get("start_date")
    end_date = conf.get("end_date")

    if not start_date or not end_date:
        raise ValueError(
            "ml_drift_monitoring_dag requires 'start_date' and 'end_date' "
            "in dag_run.conf. Trigger with: "
            "{'start_date': 'YYYY-MM-DD', 'end_date': 'YYYY-MM-DD'}"
        )

    logger.info(
        f"Triggering Evidently multi-type drift evaluation: {start_date} → {end_date} | "
        f"Run Time (Cairo): {cairo_now_str}"
    )

    from ml.monitoring import DriftMonitoringPipeline

    pipeline = DriftMonitoringPipeline()
    verdict = pipeline.run(start_date=start_date, end_date=end_date)

    logger.info(
        f"Drift evaluation completed: Severity={verdict.get('drift_severity')} | "
        f"Alert Trigger={verdict.get('alert_trigger')} | Action={verdict.get('recommended_action')}"
    )
    # Store dates in verdict for downstream XCom pull
    verdict["start_date"] = start_date
    verdict["end_date"] = end_date
    verdict["evaluated_at_cairo"] = cairo_now_str
    verdict["timezone"] = "Africa/Cairo"
    return verdict


def evaluate_drift_gate_callable(**kwargs) -> bool:
    """
    Inspect drift evaluation from upstream task.
    Returns True to proceed to retrain trigger if drift is detected,
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


def trigger_retrain_callable(**kwargs) -> dict:
    """
    Pull start_date and end_date from the drift evaluation XCom verdict and
    trigger the retrain DAG with them as conf. Uses Airflow's trigger_dag API
    so that conf values are real Python values, not Jinja templates
    (TriggerDagRunOperator.conf does not render Jinja in Airflow 2.x).
    """
    from airflow.api.common.trigger_dag import trigger_dag

    ti = kwargs["ti"]
    verdict = ti.xcom_pull(task_ids="evaluate_evidently_drift")

    start_date = verdict.get("start_date")
    end_date = verdict.get("end_date")

    logger.info(
        f"Triggering ml_model_retrain_and_evaluate_dag with "
        f"start_date={start_date}, end_date={end_date}"
    )

    trigger_dag(
        dag_id="ml_model_retrain_and_evaluate_dag",
        conf={"start_date": start_date, "end_date": end_date},
        replace_microseconds=False,
    )
    return {"triggered_dag": "ml_model_retrain_and_evaluate_dag", "start_date": start_date, "end_date": end_date}


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
    start_date=datetime(2020, 1, 1, tzinfo=cairo_tz),
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

    trigger_retrain_task = PythonOperator(
        task_id="trigger_model_retrain_dag",
        python_callable=trigger_retrain_callable,
        provide_context=True,
    )

    evaluate_drift_task >> drift_gate_task >> trigger_retrain_task