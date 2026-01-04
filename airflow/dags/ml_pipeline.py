from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime

PIPELINE_CMD = "python /opt/airflow/pipelines/run_pipeline.py"

default_args = {
    "owner": "ml",
    "retries": 1,
}

with DAG(
    dag_id="ml_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule_interval="0 3 * * 1",  # her pazartesi
    catchup=False,
    default_args=default_args,
) as dag:

    run_auto = BashOperator(
        task_id="run_auto_pipeline",
        bash_command=f"{PIPELINE_CMD} /opt/airflow/configs/train_auto.yaml",
    )

    run_auto
