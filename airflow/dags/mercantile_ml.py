"""Mercantile ML pipeline: retrain on the data lab host, then refresh the public dashboard.

The Airflow host holds an SSH key that can only run these two named actions on the data lab host.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

KEY = "/opt/airflow/dags/.ml_ssh/id_ed25519"
SSH = (
    f"ssh -i {KEY} -o BatchMode=yes -o StrictHostKeyChecking=accept-new "
    "-o UserKnownHostsFile=/tmp/ml_known_hosts root@{{ var.value.data_lab_host }}"
)

with DAG(
    dag_id="mercantile_ml_pipeline",
    default_args={"owner": "alejandro", "retries": 1, "retry_delay": timedelta(minutes=5)},
    schedule="0 7 * * *",
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,
    tags=["mercantile", "ml"],
) as dag:
    train_models = BashOperator(
        task_id="train_models",
        bash_command=f"{SSH} train",
        execution_timeout=timedelta(minutes=15),
    )
    publish_dashboard = BashOperator(
        task_id="publish_dashboard",
        bash_command=f"{SSH} publish",
        execution_timeout=timedelta(minutes=5),
    )
    train_models >> publish_dashboard
