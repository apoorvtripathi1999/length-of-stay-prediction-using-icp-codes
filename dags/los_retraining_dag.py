from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator


PROJECT_DIR = Path(os.getenv("LOS_PROJECT_DIR", Path(__file__).resolve().parents[1]))
RUN_DIR = f"{PROJECT_DIR}/runs/{{{{ run_id }}}}"
PYTHON = os.getenv("LOS_PYTHON", "python")

default_args = {
    "owner": "ml-platform",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="length_of_stay_retraining",
    description="Download, retrain, evaluate, promote, and deploy the LOS model.",
    start_date=datetime(2026, 1, 1),
    schedule="0 2 * * 0",
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["machine-learning", "length-of-stay"],
) as dag:
    download_data = BashOperator(
        task_id="download_data",
        bash_command=(
            f"mkdir -p {RUN_DIR} && cd {PROJECT_DIR} && "
            f"{PYTHON} -m ml.download_data --output {RUN_DIR}/main.csv"
        ),
        env={"NY_DATA_API_KEY": "{{ var.value.get('ny_data_api_key', '') }}"},
        append_env=True,
    )
    prepare_features = BashOperator(
        task_id="prepare_features",
        bash_command=(
            f"cd {PROJECT_DIR} && {PYTHON} -m ml.prepare_features "
            f"--input {RUN_DIR}/main.csv --output-dir {RUN_DIR}/prepared "
            f"--assets-dir {PROJECT_DIR}/data"
        ),
    )
    train_model = BashOperator(
        task_id="train_model",
        bash_command=(
            f"cd {PROJECT_DIR} && {PYTHON} -m ml.train "
            f"--input-dir {RUN_DIR}/prepared --output-dir {RUN_DIR}/candidate"
        ),
    )
    evaluate_model = BashOperator(
        task_id="evaluate_model",
        bash_command=(
            f"cd {PROJECT_DIR} && {PYTHON} -m ml.evaluate "
            f"--data-dir {RUN_DIR}/prepared --model-dir {RUN_DIR}/candidate "
            f"--output {RUN_DIR}/evaluation.json"
        ),
    )
    promote_model = BashOperator(
        task_id="promote_model",
        bash_command=(
            f"cd {PROJECT_DIR} && {PYTHON} -m ml.promote_model "
            f"--candidate-dir {RUN_DIR}/candidate --evaluation {RUN_DIR}/evaluation.json "
            f"--production-dir {PROJECT_DIR}/artifacts"
        ),
    )
    deploy_service = BashOperator(
        task_id="deploy_service",
        bash_command=f"cd {PROJECT_DIR} && docker compose up -d --build los-api",
    )
    health_check = BashOperator(
        task_id="health_check",
        bash_command="python -c \"import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')\"",
    )

    download_data >> prepare_features >> train_model >> evaluate_model >> promote_model >> deploy_service >> health_check
