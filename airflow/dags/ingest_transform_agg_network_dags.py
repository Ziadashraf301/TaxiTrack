from random import seed
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
from ingestion.logger import setup_logging
from ingestion import main_run_ingestion
from ingestion.db_utils import format_monthly_filename

def ingest_task_fn(dataset_type, **kwargs):
    
    # Init logging once per task
    logger = setup_logging()
    logger.info(f"Starting ingestion for dataset_type={dataset_type}, ds={kwargs['ds']}")

    # Airflow passes execution_date as a datetime object in the PythonOperator's context
    # We use kwargs['ds'] for the 'YYYY-MM-DD' string format consistent with dbt var
    ds_str = kwargs['ds'] 
    
    ingestion_date_obj = datetime.strptime(ds_str, '%Y-%m-%d')
    file_name = format_monthly_filename(dataset_type, ingestion_date_obj)
    
    # Call your actual ingestion logic
    main_run_ingestion.run_ingestion_for_date(file_name, ingestion_date_obj)


# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'retries': 5,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': True,
    'email': ['ziadashraf98765@gmail.com'],
    'depends_on_past': True, # Ensure tasks depend on past runs
    'start_date': datetime(2019, 1, 1), # Start date for the DAG
    'execution_timeout': timedelta(hours=1), # Timeout for each task
}

with DAG(
    default_args=default_args, 
    dag_id='ingest_transform_agg_network_dag',
    catchup=True,
    max_active_runs=1,
    schedule_interval='@monthly',
    tags=['taxi', 'ingestion', 'dbt', 'fact table', 'aggregation metrics', 'location for network analysis']
) as dag:
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
    dbt_run_stagging_models = BashOperator(
        task_id='dbt_run_stagging_models',
        bash_command=(
            f"docker exec dbt dbt run --select stg_yellow_trips stg_green_trips"
        )
    )

    dbt_run_seed = BashOperator(
        task_id='dbt_run_seed',
        bash_command=(
            f"docker exec dbt dbt seed"
        )
    )
    
    dbt_run_intermediate_models = BashOperator(
        task_id='dbt_run_intermediate_models',
        bash_command=(
            f"docker exec dbt dbt run --select stg_all_trips"
        )
    )

    mart_daily_taxi_performance = BashOperator(
        task_id='mart_daily_taxi_performance',
        bash_command=(
            f"docker exec dbt dbt run --select mart_daily_taxi_performance"
        )
    )

    mart_demand_prediction = BashOperator(
        task_id='mart_demand_prediction',
        bash_command=(
            f"docker exec dbt dbt run --select mart_demand_prediction"
        )
    )

    mart_driver_allocation = BashOperator(
        task_id='mart_driver_allocation',
        bash_command=(
            f"docker exec dbt dbt run --select mart_driver_allocation"
        )
    )

    mart_trip_location_network_metrics = BashOperator(
        task_id='mart_trip_location_network_metrics.sql',
        bash_command=(
            f"docker exec dbt dbt run --select mart_trip_location_network_metrics.sql"
        )
    )



[ingest_yellow, ingest_green]  >> dbt_run_stagging_models >> dbt_run_seed >> dbt_run_intermediate_models
dbt_run_intermediate_models >> [mart_daily_taxi_performance, mart_demand_prediction, mart_driver_allocation, mart_trip_location_network_metrics]