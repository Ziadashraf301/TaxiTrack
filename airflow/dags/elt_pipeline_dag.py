# airflow/dags/elt_pipeline_dag.py
"""
Production ELT Pipeline DAG for NYC Taxi Data (Airflow 2.10+)
Orchestrates idempotent monthly ingestion, native dbt transformations, and automated data quality gates.
Preserves internal dag_id='ingest_transform_agg_network_dag' to maintain historical run state.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from core.logging import get_logger

logger = get_logger("taxitrack_airflow_dag")


def ingest_task_callable(dataset_type: str, **kwargs):
    """Callable executing monthly ingestion for Green or Yellow taxi data with rich UI logging."""
    ds_str = kwargs["ds"]
    task_instance = kwargs.get("task_instance")
    task_id = task_instance.task_id if task_instance else f"ingest_{dataset_type}"
    run_id = kwargs.get("run_id", "manual")

    execution_date = datetime.strptime(ds_str, "%Y-%m-%d")
    logger.info(
        f"\n====================================================================\n"
        f"  AIRFLOW TASK: {task_id}\n"
        f"  DATASET:      {dataset_type.upper()}\n"
        f"  PARTITION:    {ds_str} ({execution_date.strftime('%B %Y')})\n"
        f"  RUN ID:       {run_id}\n"
        f"===================================================================="
    )

    start_time = datetime.now()
    try:
        from data.pipeline import run_monthly_ingestion
        result = run_monthly_ingestion(dataset_type=dataset_type, execution_date=execution_date)
        duration = (datetime.now() - start_time).total_seconds()
        status = result.get("status", "unknown").upper()
        rows = result.get("rows", 0)
        file_name = result.get("file_name", "")

        logger.info(
            f"\n====================================================================\n"
            f"  TASK COMPLETED: {task_id}\n"
            f"  STATUS:         {status}\n"
            f"  FILE:           {file_name}\n"
            f"  ROWS INSERTED:  {rows:,}\n"
            f"  DURATION:       {duration:.2f}s\n"
            f"===================================================================="
        )
        return result
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(
            f"\n====================================================================\n"
            f"  TASK FAILED:    {task_id}\n"
            f"  DURATION:       {duration:.2f}s\n"
            f"  ERROR:          {e}\n"
            f"====================================================================",
            exc_info=True
        )
        raise


def create_dbt_run_task(model_name: str) -> BashOperator:
    """Helper to generate native dbt run task without Docker-in-Docker."""
    clean_name = model_name.replace(".sql", "")
    return BashOperator(
        task_id=f"dbt_run_{clean_name}",
        bash_command=f"dbt run --select {clean_name} --project-dir /opt/dbt --profiles-dir /opt/dbt --log-path /tmp/dbt_logs --target-path /tmp/dbt_target"
    )


default_args = {
    "owner": "airflow",
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "depends_on_past": False,  # Allows testing any single month independently without prior month dependencies
    "start_date": datetime(2019, 1, 1),
    "execution_timeout": timedelta(hours=1),
}

with DAG(
    dag_id="ingest_transform_agg_network_dag",  # Matches historical Postgres cluster runs
    default_args=default_args,
    schedule_interval="@monthly",
    catchup=False,  # Set to False so Airflow doesn't queue 80 historical months at startup!
    max_active_runs=1,
    tags=["taxi", "ingestion", "dbt", "fact table", "aggregation", "ml", "network analysis"],
) as dag:

    # -------------------------------------------------------------------------
    # Ingestion Stage (Direct Streaming MinIO S3 -> ClickHouse Batch Tables)
    # -------------------------------------------------------------------------
    ingest_green = PythonOperator(
        task_id="ingest_monthly_green_tripdata",
        python_callable=ingest_task_callable,
        op_kwargs={"dataset_type": "green"},
    )

    ingest_yellow = PythonOperator(
        task_id="ingest_monthly_yellow_tripdata",
        python_callable=ingest_task_callable,
        op_kwargs={"dataset_type": "yellow"},
    )

    # -------------------------------------------------------------------------
    # dbt Staging (Seeds are static and already loaded)
    # -------------------------------------------------------------------------
    dbt_run_stg_green = create_dbt_run_task("stg_green_trips")
    dbt_run_stg_yellow = create_dbt_run_task("stg_yellow_trips")

    dbt_run_int_all_trips = create_dbt_run_task("int_all_trips")

    # -------------------------------------------------------------------------
    # dbt Dimensional Analytics & Feature Marts
    # -------------------------------------------------------------------------
    mart_models = [
        "mart_daily_taxi_performance",
        "mart_demand_prediction",
        "mart_trip_location_network_metrics",
    ]
    mart_tasks = [create_dbt_run_task(model) for model in mart_models]

    # -------------------------------------------------------------------------
    # Pipeline Dependencies
    # -------------------------------------------------------------------------
    ingest_green >> dbt_run_stg_green
    ingest_yellow >> dbt_run_stg_yellow

    [dbt_run_stg_green, dbt_run_stg_yellow] >> dbt_run_int_all_trips
    dbt_run_int_all_trips >> mart_tasks
