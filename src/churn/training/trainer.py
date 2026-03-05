"""
Training pipeline orchestrator.
Runs the full end-to-end training lifecycle:
    1. Load materialised feature matrix
    2. Temporal split
    3. Optuna hyperparameter tuning
    4. Train LightGBM with best params
    5. Calibrate probabilities
    6. Evaluate on test set
    7. Log all metrics + artefacts to MLflow
    8. Save model to outputs/model_artefacts/

Entry point: python -m churn.training.trainer
         or: make train
"""
from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import mlflow.lightgbm
import pandas as pd

from churn.config import cfg
from churn.data.splitter import temporal_split, assert_no_leakage
from churn.models.lgbm_model import LGBMChurnModel, EXCLUDE_COLS
from churn.models.calibrator import ChurnCalibrator
from churn.training.tuner import run_study

logger = logging.getLogger(__name__)


def _load_feature_matrix(features_dir: Path) -> pd.DataFrame:
    """Load and concatenate all materialised feature Parquet files."""
    parquet_files = sorted(features_dir.glob("features_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No feature parquet files found in {features_dir}. "
            "Run `make features` first."
        )
    dfs = [pd.read_parquet(p) for p in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    logger.info("Loaded %s feature files → %s rows", len(parquet_files), f"{len(df):,}")
    return df


def train(
    n_trials: int = 50,
    skip_tuning: bool = False,
) -> tuple[LGBMChurnModel, ChurnCalibrator]:
    """
    Full training pipeline.

    Args:
        n_trials    : Number of Optuna trials for hyperparameter search
        skip_tuning : If True, use default params (faster for debugging)

    Returns:
        (model, calibrator) — fitted and calibrated
    """
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run() as run:
        logger.info("MLflow run ID: %s", run.info.run_id)

        # ── 1. Load features ──────────────────────────────────────────────────
        df = _load_feature_matrix(cfg.paths.features_dir)

        # ── 2. Temporal split ─────────────────────────────────────────────────
        split = temporal_split(df, date_col="snapshot_date")
        assert_no_leakage(split.train, split.test, date_col="snapshot_date")

        X_train, y_train = split.train, split.train["churned"]
        X_val,   y_val   = split.val,   split.val["churned"]
        X_test,  y_test  = split.test,  split.test["churned"]

        mlflow.log_params({
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "train_churn_rate": round(float(y_train.mean()), 4),
            "val_churn_rate": round(float(y_val.mean()), 4),
            "test_churn_rate": round(float(y_test.mean()), 4),
            "prediction_window_days": cfg.label.prediction_window_days,
        })

        # ── 3. Hyperparameter tuning ──────────────────────────────────────────
        if skip_tuning:
            best_params = {}
            logger.info("Skipping Optuna tuning — using default params")
        else:
            best_params = run_study(
                X_train, y_train, X_val, y_val, n_trials=n_trials
            )
            mlflow.log_params({f"lgbm_{k}": v for k, v in best_params.items()})

        # ── 4. Train final model ──────────────────────────────────────────────
        model = LGBMChurnModel(params=best_params)
        model.fit(X_train, y_train, X_val, y_val)
        mlflow.log_param("best_iteration", model.booster.best_iteration)

        # ── 5. Calibrate on validation set ────────────────────────────────────
        calibrator = ChurnCalibrator(method="isotonic")
        calibrator.fit(model, X_val, y_val)

        # ── 6. Evaluate on test set ───────────────────────────────────────────
        from churn.evaluation.metrics import compute_all_metrics, log_metrics_to_mlflow

        cal_probs = calibrator.predict_proba(X_test)
        metrics = compute_all_metrics(y_test, cal_probs)
        log_metrics_to_mlflow(metrics)

        logger.info(
            "Test set — AUC-PR=%.4f  AUC-ROC=%.4f  F1=%.4f",
            metrics["auc_pr"], metrics["auc_roc"], metrics["f1"],
        )

        # ── 7. Save artefacts ─────────────────────────────────────────────────
        artefact_dir = cfg.paths.model_artefacts_dir
        artefact_dir.mkdir(parents=True, exist_ok=True)

        model_path = artefact_dir / "lgbm_churn_model.pkl"
        cal_path   = artefact_dir / "calibrator.pkl"
        model.save(model_path)
        calibrator.save(cal_path)

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(cal_path))

        # Register model in MLflow model registry
        mlflow.lightgbm.log_model(
            lgb_model=model.booster,
            artifact_path="lgbm_model",
            registered_model_name=cfg.model.registry_name,
        )

        logger.info(
            "Training complete — run_id=%s  model saved → %s",
            run.info.run_id, model_path,
        )

    return model, calibrator


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train()
