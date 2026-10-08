import os
import sys
from datetime import datetime
from airflow import DAG

try:
    from airflow.providers.standard.operators.python import PythonOperator
except ImportError:
    from airflow.operators.python import PythonOperator

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_preprocessing import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model
from src.deploy import deploy_model

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
}

with DAG(
    dag_id="ml_pipeline_dag",
    default_args=default_args,
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops", "ci-cd", "iris"]
) as dag:

    preprocess_data_task = PythonOperator(
        task_id="preprocess_data_task",
        python_callable=preprocess_data
    )

    train_model_task = PythonOperator(
        task_id="train_model_task",
        python_callable=train_model
    )

    evaluate_model_task = PythonOperator(
        task_id="evaluate_model_task",
        python_callable=evaluate_model
    )

    deploy_model_task = PythonOperator(
        task_id="deploy_model_task",
        python_callable=deploy_model
    )

    preprocess_data_task >> train_model_task >> evaluate_model_task >> deploy_model_task
