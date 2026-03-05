"""
LightGBM churn classifier.
Primary model — chosen for:
  - Leaf-wise tree growth (better accuracy on tabular data vs. level-wise)
  - Native categorical feature handling
  - Native missing-value support (no imputation required)
  - is_unbalance flag handles class imbalance without external resampling
  - Speed — essential for Optuna hyperparameter search
"""

from __future__ import annotations

import logging
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from churn.config import PROJECT_ROOT
from churn.models.base_model import BaseChurnModel

logger = logging.getLogger(__name__)

LGBM_CONF_PATH = PROJECT_ROOT / "conf" / "model" / "lightgbm.yaml"

# Features to exclude from model input (identifiers, date cols, raw categoricals)
EXCLUDE_COLS = {
    "unique_customer_identifier",
    "snapshot_date",
    "churned",
    "tenure_bucket",  # use tenure_log instead
    "sales_channel",  # use sales_channel_encoded
    "crm_package_name",  # use crm_package_name_encoded
}


class LGBMChurnModel(BaseChurnModel):
    """
    Gradient-boosted binary classifier wrapping LightGBM.
    Hyperparameters loaded from conf/model/lightgbm.yaml — overridden
    by Optuna best trial at training time.
    """

    def __init__(self, params: dict | None = None) -> None:
        """
        Args:
            params : Override default LightGBM params. If None, loads from
                     conf/model/lightgbm.yaml.
        """
        raw = OmegaConf.load(LGBM_CONF_PATH)
        raw_container = OmegaConf.to_container(raw, resolve=True)
        if not isinstance(raw_container, dict):
            raise TypeError("Expected LightGBM config to be a mapping")
        defaults = {k: v for k, v in raw_container.items() if k != "optuna"}
        self.params = {**defaults, **(params or {})}
        self._booster: lgb.Booster | None = None
        self._feature_names: list[str] = []

    def _get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[[c for c in df.columns if c not in EXCLUDE_COLS]]

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> LGBMChurnModel:
        X_tr = self._get_features(X_train)
        self._feature_names = list(X_tr.columns)

        train_data = lgb.Dataset(X_tr, label=y_train, feature_name=self._feature_names)
        valid_sets = [train_data]
        valid_names = ["train"]
        callbacks: list[Callable[..., Any]] = [lgb.log_evaluation(period=100)]

        if X_val is not None and y_val is not None:
            X_v = self._get_features(X_val)
            val_data = lgb.Dataset(X_v, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append("val")
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=self.params.get("early_stopping_rounds", 50),
                    verbose=False,
                )
            )

        n_estimators = self.params.pop("n_estimators", 1000)
        early_stop = self.params.pop("early_stopping_rounds", 50)

        self._booster = lgb.train(
            params=self.params,
            train_set=train_data,
            num_boost_round=n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        # Restore for serialisation
        self.params["n_estimators"] = n_estimators
        self.params["early_stopping_rounds"] = early_stop

        logger.info(
            "LGBMChurnModel trained — best_iteration=%s",
            self._booster.best_iteration,
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._booster is None:
            raise RuntimeError("Model has not been trained. Call .fit() first.")
        X_feat = self._get_features(X).copy()

        model_feature_names = self._feature_names or list(self._booster.feature_name())
        if model_feature_names:
            missing = [c for c in model_feature_names if c not in X_feat.columns]
            for col in missing:
                X_feat[col] = 0.0
            X_feat = X_feat[model_feature_names]

        preds = self._booster.predict(
            X_feat,
            num_iteration=self._booster.best_iteration,
        )
        return np.asarray(preds, dtype=float)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)
        logger.info("Model saved → %s", path)

    @classmethod
    def load(cls, path: Path) -> LGBMChurnModel:
        try:
            with path.open("rb") as f:
                model = pickle.load(f)
        except Exception:
            model = joblib.load(path)
        logger.info("Model loaded ← %s", path)
        return cast(LGBMChurnModel, model)

    @property
    def booster(self) -> lgb.Booster:
        if self._booster is None:
            raise RuntimeError("Model not trained.")
        return self._booster
