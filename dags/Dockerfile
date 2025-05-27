# TaxiTrack/Dockerfile
FROM apache/airflow:2.10.2

# switch to airflow user
USER airflow

# mount the dags directory
COPY dags/requirements.txt /tmp/

# install the requirements
RUN pip install --no-cache-dir -r /tmp/requirements.txt