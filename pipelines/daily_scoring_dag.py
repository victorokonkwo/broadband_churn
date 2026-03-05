"""
Airflow DAG — nightly batch scoring.
Runs every night at 02:00 UTC to produce a fresh scored_customers.csv
that the retention team picks up in the morning.

Prerequisites:
    - Model artefacts exist in outputs/model_artefacts/
    - DuckDB has been refreshed with latest data (separate ingest DAG)
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


default_args = {
    "owner": "data-science",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": True,
    "email": ["ds-team@uktelecom.co.uk"],
}


def _ingest_data():
    from churn.data.loader import ingest_raw_tables
    ingest_raw_tables()


def _build_features():
    from churn.features.feature_store import materialise_features
    from datetime import date
    materialise_features([str(date.today())])


def _score_customers():
    from churn.scoring.batch_scorer import score_active_customers
    score_active_customers()


def _run_drift_check():
    from churn.monitoring.drift_detector import run_full_monitoring
    import pandas as pd
    from churn.config import cfg
    features_dir = cfg.paths.features_dir
    parquets = sorted(features_dir.glob("features_*.parquet"))
    if len(parquets) >= 2:
        ref = pd.read_parquet(parquets[0])
        cur = pd.read_parquet(parquets[-1])
        result = run_full_monitoring(ref, cur)
        if result["drift_detected"]:
            from churn.monitoring.alerting import format_drift_alert, send_slack_alert
            title, msg, sev = format_drift_alert(result)
            # send_slack_alert(webhook_url=..., title=title, message=msg, severity=sev)


with DAG(
    dag_id="churn_daily_scoring",
    schedule_interval="0 2 * * *",   # daily at 02:00 UTC
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["churn", "scoring"],
) as dag:

    ingest = PythonOperator(task_id="ingest_raw_data", python_callable=_ingest_data)
    features = PythonOperator(task_id="build_features", python_callable=_build_features)
    score = PythonOperator(task_id="score_active_customers", python_callable=_score_customers)
    drift = PythonOperator(task_id="drift_check", python_callable=_run_drift_check)

    ingest >> features >> score >> drift
