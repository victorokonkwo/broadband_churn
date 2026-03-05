"""
Optuna hyperparameter tuning for LightGBM.
Optimises Average Precision (AUC-PR) on the validation set.
Runs 50 trials by default — each trial fits a LightGBM model with
early stopping, so poor trials are pruned quickly.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import lightgbm as lgb
import optuna
import pandas as pd
from omegaconf import OmegaConf
from sklearn.metrics import average_precision_score

from churn.config import PROJECT_ROOT
from churn.models.lgbm_model import EXCLUDE_COLS

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

LGBM_CONF_PATH = PROJECT_ROOT / "conf" / "model" / "lightgbm.yaml"


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def build_objective(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Callable[[optuna.Trial], float]:
    """
    Build an Optuna objective function.
    Closure captures training and validation data.
    """
    raw = OmegaConf.load(LGBM_CONF_PATH)
    search = raw.optuna.search_space
    base_params = {
        "objective": raw.objective,
        "metric": raw.metric,
        "verbose": -1,
        "n_jobs": -1,
        "is_unbalance": raw.is_unbalance,
        "feature_pre_filter": False,
    }

    feat_cols = _get_feature_cols(X_train)
    train_data = lgb.Dataset(X_train[feat_cols], label=y_train, free_raw_data=False)
    val_data = lgb.Dataset(X_val[feat_cols], label=y_val, reference=train_data, free_raw_data=False)

    def objective(trial: optuna.Trial) -> float:
        params = {
            **base_params,
            "num_leaves": trial.suggest_int("num_leaves", *search.num_leaves),
            "learning_rate": trial.suggest_float("learning_rate", *search.learning_rate, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", *search.min_child_samples),
            "feature_fraction": trial.suggest_float("feature_fraction", *search.feature_fraction),
            "bagging_fraction": trial.suggest_float("bagging_fraction", *search.bagging_fraction),
            "bagging_freq": 5,
            "reg_alpha": trial.suggest_float("reg_alpha", *search.reg_alpha),
            "reg_lambda": trial.suggest_float("reg_lambda", *search.reg_lambda),
        }

        pruning_callback = optuna.integration.LightGBMPruningCallback(
            trial, "average_precision", valid_name="val"
        )

        booster = lgb.train(
            params=params,
            train_set=train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(-1),
                pruning_callback,
            ],
        )

        preds = booster.predict(X_val[feat_cols], num_iteration=booster.best_iteration)
        score = average_precision_score(y_val, preds)
        return float(score)

    return objective


def run_study(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
    study_name: str = "churn-lgbm",
) -> dict[str, Any]:
    """
    Run Optuna study and return best hyperparameters.

    Returns:
        dict of best hyperparameters to pass into LGBMChurnModel(params=...)
    """
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
    )

    objective = build_objective(X_train, y_train, X_val, y_val)

    logger.info("Starting Optuna study — %s trials, maximising AUC-PR …", n_trials)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(
        "Best trial: AUC-PR=%.4f  params=%s",
        study.best_value,
        study.best_params,
    )

    return study.best_params


# Alias for notebook imports
run_optuna_study = run_study
