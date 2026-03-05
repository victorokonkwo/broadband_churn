"""
Cross-validated target encoding.
Replaces high-cardinality categoricals (crm_package_name, sales_channel)
with the mean churn rate for that category — using K-fold cross-validation
to prevent target leakage within the training set.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


class CrossValidatedTargetEncoder:
    """
    Mean target encoder with cross-validation to prevent in-fold leakage.

    For each fold:
        - Compute category mean of target on the OTHER folds
        - Apply to the current fold

    At inference time: use global means computed on the full training set.

    Args:
        columns    : Columns to encode
        n_splits   : Number of CV folds (default 5)
        smoothing  : Additive smoothing strength — blends category mean towards
                     global mean when category has few observations (λ=1 default)
        target_col : Name of the binary target column
    """

    def __init__(
        self,
        columns: Sequence[str],
        n_splits: int = 5,
        smoothing: float = 1.0,
        target_col: str = "churned",
    ) -> None:
        self.columns = list(columns)
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.target_col = target_col
        self._global_mean: float = 0.0
        self._category_means: dict[str, dict] = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit on training data and transform using out-of-fold means.
        Call this on the training set only.
        """
        df = df.copy()
        self._global_mean = df[self.target_col].mean()

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        indices = np.arange(len(df))

        for col in self.columns:
            encoded = np.full(len(df), self._global_mean)

            for train_idx, val_idx in kf.split(indices):
                fold_train = df.iloc[train_idx]
                cat_means = (
                    fold_train.groupby(col)[self.target_col]
                    .agg(["sum", "count"])
                    .rename(columns={"sum": "n_pos", "count": "n"})
                )
                cat_means["smoothed_mean"] = (
                    (cat_means["n_pos"] + self.smoothing * self._global_mean)
                    / (cat_means["n"] + self.smoothing)
                )
                mapping = cat_means["smoothed_mean"].to_dict()
                encoded[val_idx] = (
                    df.iloc[val_idx][col]
                    .map(mapping)
                    .fillna(self._global_mean)
                    .to_numpy()
                )

            df[f"{col}_encoded"] = encoded

            # Fit global means for inference
            cat_stats = (
                df.groupby(col)[self.target_col]
                .agg(["sum", "count"])
                .rename(columns={"sum": "n_pos", "count": "n"})
            )
            cat_stats["smoothed_mean"] = (
                (cat_stats["n_pos"] + self.smoothing * self._global_mean)
                / (cat_stats["n"] + self.smoothing)
            )
            self._category_means[col] = cat_stats["smoothed_mean"].to_dict()
            logger.info("  Encoded %s (%s categories)", col, len(self._category_means[col]))

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply pre-fitted category means to a new DataFrame (val / test / scoring)."""
        df = df.copy()
        for col in self.columns:
            df[f"{col}_encoded"] = (
                df[col]
                .map(self._category_means.get(col, {}))
                .fillna(self._global_mean)
            )
        return df
