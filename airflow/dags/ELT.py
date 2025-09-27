from datetime import datetime, timedelta, date
import calendar
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from ingestion.logger import setup_logging
from ingestion import main_run_ingestion
from ingestion.db_utils import format_monthly_filename
import sys
import os
from pathlib import Path

# Add demand_prediction/model_dev to sys.path
sys.path.append("/opt/airflow/demand_prediction/model_dev")

# Import your pipeline
from pipeline import TimeSeriesForecastPipeline
from config import CLICKHOUSE_CONFIG

# Initialize logger once
logger = setup_logging()

def ingest_task_fn(dataset_type, **kwargs):
    ds_str = kwargs['ds']
    ingestion_date_obj = datetime.strptime(ds_str, '%Y-%m-%d')
    file_name = format_monthly_filename(dataset_type, ingestion_date_obj)

    logger.info(f"Starting ingestion for dataset_type={dataset_type}, ds={ds_str}")
    main_run_ingestion.run_ingestion_for_date(file_name, ingestion_date_obj)


def run_pipeline_task_fn(**kwargs):
    # import calendar
    # from datetime import datetime, date

    # ds_str = kwargs['ds']
    # exec_date = datetime.strptime(ds_str, "%Y-%m-%d")

    # # Month start & end (only that month’s window)
    # # start_date = exec_date.replace(day=1).date()
    # start_date = date(2019,1,1)
    # last_day = calendar.monthrange(exec_date.year, exec_date.month)[1]
    # end_date = exec_date.replace(day=last_day).date()

    # logger.info(f"Running TimeSeriesForecastPipeline for {start_date} → {end_date}")

    # pipeline = TimeSeriesForecastPipeline()
    # pipeline.run_pipeline(
    #     table_name=CLICKHOUSE_CONFIG["default_table"],
    #     start_date=start_date,
    #     end_date=end_date,
    #     train= True
    # )

    logger.info("✅ Pipeline run completed!")


# Default DAG arguments
default_args = {
    'owner': 'airflow',
    'retries': 5,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': True,
    'email': ['ziadashraf98765@gmail.com'],
    'depends_on_past': True,
    'start_date': datetime(2019, 1, 1),
    'execution_timeout': timedelta(hours=1),
}


# Helper to create dbt BashOperator dynamically
def create_dbt_task(model_name):
    return BashOperator(
        task_id=f"dbt_run_{model_name.replace('.sql','')}",
        bash_command=f"docker exec dbt dbt run --select {model_name}"
    )


with DAG(
    dag_id='ingest_transform_agg_network_dag',
    default_args=default_args,
    schedule_interval='@monthly',
    catchup=True,
    max_active_runs=1,
    tags=['taxi', 'ingestion', 'dbt', 'fact table', 'aggregation', 'ml', 'network analysis'],
) as dag:

    # Ingestion tasks
    ingest_green = PythonOperator(
        task_id='ingest_monthly_green_tripdata',
        python_callable=ingest_task_fn,
        op_kwargs={'dataset_type': 'green_tripdata'},
    )
    ingest_yellow = PythonOperator(
        task_id='ingest_monthly_yellow_tripdata',
        python_callable=ingest_task_fn,
        op_kwargs={'dataset_type': 'yellow_tripdata'},
    )

    dbt_run_seed = BashOperator(
        task_id='dbt_run_seed',
        bash_command="docker exec dbt dbt seed"
    )

    # dbt tasks
    dbt_run_stg_yellow = create_dbt_task("stg_yellow_trips")
    dbt_run_stg_green = create_dbt_task("stg_green_trips")
    dbt_run_intermediate = create_dbt_task("stg_all_trips")

    mart_models = [
        "mart_daily_taxi_performance",
        "mart_demand_prediction",
        "mart_trip_location_network_metrics.sql"
    ]
    mart_tasks = [create_dbt_task(m) for m in mart_models]

    run_forecast_pipeline = PythonOperator(
        task_id="run_forecast_pipeline",
        python_callable=run_pipeline_task_fn,
        provide_context=True
    )

    # Dependencies
    ingest_yellow >> dbt_run_stg_yellow
    ingest_green >> dbt_run_stg_green

    for stg_task in [dbt_run_stg_yellow, dbt_run_stg_green]:
        stg_task >> dbt_run_seed

    dbt_run_seed >> dbt_run_intermediate

    for mart_task in mart_tasks:
        dbt_run_intermediate >> mart_task

    # Run pipeline after marts are built
    mart_tasks >> run_forecast_pipeline
