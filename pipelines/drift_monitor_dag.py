"""
Airflow DAG — drift monitoring.
Runs every 6 hours to detect feature & prediction drift using Evidently + PSI.
Sends Slack alerts when thresholds are breached.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


default_args = {
    "owner": "data-science",
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
    "email_on_failure": True,
    "email": ["ds-team@uktelecom.co.uk"],
}


def _load_reference_and_current():
    """Load reference (training) and current (latest scored) feature matrices."""
    import pandas as pd
    from pathlib import Path
    from churn.config import cfg

    features_dir = Path(cfg.paths.features_dir)
    ref_path = features_dir / "features_train.parquet"
    cur_path = features_dir / "features_latest.parquet"

    if not ref_path.exists() or not cur_path.exists():
        raise FileNotFoundError(
            f"Missing reference or current features at {features_dir}"
        )

    ref = pd.read_parquet(ref_path)
    cur = pd.read_parquet(cur_path)
    return ref, cur


def _run_evidently_drift(**context):
    """Run Evidently drift report and push result to XCom."""
    ref, cur = _load_reference_and_current()

    from churn.monitoring.drift_detector import run_full_monitoring

    result = run_full_monitoring(ref, cur)
    context["ti"].xcom_push(key="drift_result", value=result)


def _send_alerts_if_needed(**context):
    """Format and send Slack alert when drift is detected."""
    result = context["ti"].xcom_pull(
        task_ids="run_evidently_drift", key="drift_result"
    )
    if result is None:
        return

    if result.get("drift_detected") or result.get("psi_alerts"):
        from churn.monitoring.alerting import format_drift_alert

        title, message, severity = format_drift_alert(result)
        # Uncomment and configure webhook for production
        # from churn.monitoring.alerting import send_slack_alert
        # send_slack_alert(
        #     webhook_url=os.environ["SLACK_WEBHOOK_URL"],
        #     title=title,
        #     message=message,
        #     severity=severity,
        # )


def _save_monitoring_artefacts(**context):
    """Persist monitoring results for audit trail."""
    import json
    from pathlib import Path
    from churn.config import cfg

    result = context["ti"].xcom_pull(
        task_ids="run_evidently_drift", key="drift_result"
    )
    if result is None:
        return

    reports_dir = Path(cfg.paths.outputs_dir) / "monitoring_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = reports_dir / f"drift_report_{timestamp}.json"

    # Serialise only JSON-safe keys
    safe_result = {
        k: v
        for k, v in result.items()
        if isinstance(v, (str, int, float, bool, list, dict, type(None)))
    }
    with open(out_path, "w") as f:
        json.dump(safe_result, f, indent=2, default=str)


with DAG(
    dag_id="churn_drift_monitor",
    schedule_interval="0 */6 * * *",   # every 6 hours
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["churn", "monitoring", "drift"],
) as dag:

    drift = PythonOperator(
        task_id="run_evidently_drift",
        python_callable=_run_evidently_drift,
    )
    alert = PythonOperator(
        task_id="send_alerts_if_needed",
        python_callable=_send_alerts_if_needed,
    )
    save = PythonOperator(
        task_id="save_monitoring_artefacts",
        python_callable=_save_monitoring_artefacts,
    )

    drift >> [alert, save]
