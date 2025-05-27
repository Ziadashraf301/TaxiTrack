from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import json # Not directly used in the bash_command after this fix, but kept if other PythonOperators need it
import pendulum # It's good practice to use pendulum for start_date with Airflow 2+

# Assuming ingestion.main is a local module
# from ingestion.main import run_ingestion_for_date

default_args = {
    'owner': 'airflow',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': True,
    'email': ['ziadashraf98765@gmail.com'],
}

DBT_SERVICE_NAME = 'dbt'  # Matches service name in docker-compose.yml

def format_monthly_filename(execution_date):
    return f"green_tripdata_{execution_date.strftime('%Y-%m')}.parquet"

# Placeholder for run_ingestion_for_date - ensure this function exists and is imported correctly
# If run_ingestion_for_date is not defined or imported, this will cause an error
def run_ingestion_for_date(file_name):
    print(f"Simulating ingestion for file: {file_name}")
    # In a real scenario, this would call your actual ingestion logic

def ingest_task_fn(execution_date, **kwargs):
    # Airflow passes execution_date as a datetime object in the PythonOperator's context
    # We use kwargs['ds'] for the 'YYYY-MM-DD' string format consistent with dbt var
    ds_str = kwargs['ds'] 
    
    ingestion_date_obj = datetime.strptime(ds_str, '%Y-%m-%d')
    file_name = format_monthly_filename(ingestion_date_obj)
    
    # Call your actual ingestion logic
    run_ingestion_for_date(file_name)


with DAG(
    dag_id='monthly_taxi_ingestion_with_dbt',
    default_args=default_args,
    description='Ingest NYC green taxi data monthly and run dbt transformations',
    start_date=pendulum.datetime(2019, 1, 1, tz="UTC"), # Use pendulum for timezone-aware start_date
    schedule_interval='@monthly',
    catchup=True,
    max_active_runs=1,
    tags=['taxi', 'ingestion', 'dbt'],
    # user_defined_macros are removed as we are not using the 'to_json' filter directly in bash_command
) as dag:

    ingest_data = PythonOperator(
        task_id='ingest_monthly_green_tripdata',
        python_callable=ingest_task_fn,
        provide_context=True, # Essential for passing Airflow context variables like 'ds'
    )

    dbt_deps = BashOperator(
        task_id='dbt_install_dependencies',
        bash_command=f"docker exec {DBT_SERVICE_NAME} dbt deps",
        do_xcom_push=False,
    )

    dbt_comp = BashOperator(
        task_id='dbt_compile',
        bash_command=f"docker exec {DBT_SERVICE_NAME} dbt compile",
        do_xcom_push=False,
    )

    dbt_run = BashOperator(
        task_id='dbt_run_models',
        bash_command=(
            f"docker exec {DBT_SERVICE_NAME} dbt run "
            # Direct string formatting for --vars.
            # {{ ds }} is Airflow's Jinja for the execution date.
            # The outer ' ' are for the bash command argument.
            # The inner \" \" are for the JSON string value.
            # The doubled {{ and }} escape literal curly braces for the f-string.
            f"--vars '{{\"execution_date\": \"{{{{ ds }}}}\"}}' " 
        ),
        do_xcom_push=False,
    )

    ingest_data >> dbt_deps >> dbt_comp >> dbt_run