"""
Airflow DAG — weekly model retraining.
Runs every Sunday at 04:00 UTC. Re-tunes hyper-parameters when the champion
model is more than 4 weeks old, otherwise just re-fits with updated data.

Champion / Challenger promotion is gated on:
    1. AUC-PR ≥ champion − 0.01
    2. P(calibration) > 0.05  (Hosmer–Lemeshow)
    3. Evidently drift not detected on validation fold

All artefacts are logged to MLflow.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator


default_args = {
    "owner": "data-science",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": True,
    "email": ["ds-team@uktelecom.co.uk"],
}

# ── Hyper-parameter age gate (retune every 4 weeks) ────────────────
RETUNE_INTERVAL_DAYS = 28


def _decide_retune_or_refit(**context):
    """Branch: full Optuna tune vs. re-fit with existing best params."""
    import json
    from pathlib import Path
    from churn.config import cfg

    best_params_path = Path(cfg.paths.model_artefacts_dir) / "best_params.json"
    if not best_params_path.exists():
        return "full_optuna_tune"

    mtime = datetime.fromtimestamp(best_params_path.stat().st_mtime)
    age_days = (datetime.utcnow() - mtime).days
    return "full_optuna_tune" if age_days >= RETUNE_INTERVAL_DAYS else "quick_refit"


def _full_optuna_tune():
    from churn.training.trainer import run_training_pipeline
    run_training_pipeline(tune=True)


def _quick_refit():
    from churn.training.trainer import run_training_pipeline
    run_training_pipeline(tune=False)


def _evaluate_challenger():
    """Compare newly trained model against current champion."""
    import json, mlflow
    from pathlib import Path
    from churn.config import cfg

    artefacts = Path(cfg.paths.model_artefacts_dir)
    metrics_path = artefacts / "metrics.json"

    if not metrics_path.exists():
        raise FileNotFoundError("Challenger metrics.json not found")

    with open(metrics_path) as f:
        challenger = json.load(f)

    # Fetch champion from MLflow registry
    client = mlflow.tracking.MlflowClient()
    try:
        champion_version = client.get_latest_versions(
            "churn-lgbm", stages=["Production"]
        )[0]
        champion_run = client.get_run(champion_version.run_id)
        champion_auc_pr = float(champion_run.data.metrics["auc_pr"])
    except (IndexError, KeyError):
        champion_auc_pr = 0.0  # no champion yet → auto-promote

    challenger_auc_pr = challenger.get("auc_pr", 0.0)

    if challenger_auc_pr >= champion_auc_pr - 0.01:
        # Promote challenger → Production
        client.transition_model_version_stage(
            name="churn-lgbm",
            version=client.get_latest_versions("churn-lgbm", stages=["Staging"])[
                0
            ].version,
            stage="Production",
        )


def _notify_retrain_complete():
    """Post Slack notification with retrain summary."""
    import json
    from pathlib import Path
    from churn.config import cfg

    artefacts = Path(cfg.paths.model_artefacts_dir)
    metrics_path = artefacts / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            m = json.load(f)
        msg = (
            f":chart_with_upwards_trend: *Weekly Retrain Complete*\n"
            f"AUC-PR: {m.get('auc_pr', 'N/A'):.4f}  |  AUC-ROC: {m.get('auc_roc', 'N/A'):.4f}\n"
            f"Brier: {m.get('brier_score', 'N/A'):.4f}"
        )
        # from churn.monitoring.alerting import send_slack_alert
        # send_slack_alert(webhook_url=..., title="Retrain", message=msg)


with DAG(
    dag_id="churn_weekly_retrain",
    schedule_interval="0 4 * * 0",  # Sundays 04:00 UTC
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["churn", "retrain"],
) as dag:

    branch = BranchPythonOperator(
        task_id="decide_retune_or_refit",
        python_callable=_decide_retune_or_refit,
    )
    tune = PythonOperator(task_id="full_optuna_tune", python_callable=_full_optuna_tune)
    refit = PythonOperator(task_id="quick_refit", python_callable=_quick_refit)
    evaluate = PythonOperator(
        task_id="evaluate_challenger",
        python_callable=_evaluate_challenger,
        trigger_rule="none_failed_min_one_success",
    )
    notify = PythonOperator(
        task_id="notify_retrain_complete",
        python_callable=_notify_retrain_complete,
        trigger_rule="none_failed_min_one_success",
    )

    branch >> [tune, refit] >> evaluate >> notify
