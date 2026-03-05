"""
Temporal train / validation / test split.
NEVER use random splits on time-series data — this inflates AUC by 0.05–0.10.
All split boundaries are driven by conf/config.yaml → splits section.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from churn.config import cfg

logger = logging.getLogger(__name__)


@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    test_end: pd.Timestamp

    # ── convenience properties expected by modelling notebooks ─────────────
    @property
    def _feature_cols(self) -> list[str]:
        exclude = {"churned", "snapshot_date", "unique_customer_identifier"}
        return [c for c in self.train.columns if c not in exclude]

    @property
    def X_train(self) -> pd.DataFrame:
        return self.train[self._feature_cols]

    @property
    def y_train(self) -> pd.Series:
        return self.train["churned"]

    @property
    def X_val(self) -> pd.DataFrame:
        return self.val[self._feature_cols]

    @property
    def y_val(self) -> pd.Series:
        return self.val["churned"]

    @property
    def X_test(self) -> pd.DataFrame:
        return self.test[self._feature_cols]

    @property
    def y_test(self) -> pd.Series:
        return self.test["churned"]

    def log_sizes(self) -> None:
        total = len(self.train) + len(self.val) + len(self.test)
        for name, df in [("train", self.train), ("val", self.val), ("test", self.test)]:
            churn_rate = df["churned"].mean() if "churned" in df.columns else float("nan")
            logger.info(
                "  %-6s  %7s rows  churn_rate=%.3f",
                name,
                f"{len(df):,}",
                churn_rate,
            )
        logger.info("  Total: %s rows", f"{total:,}")


def temporal_split(
    df: pd.DataFrame,
    date_col: str = "snapshot_date",
) -> SplitResult:
    """
    Split df into train / val / test based on config date boundaries.

    Boundaries (from conf/config.yaml):
        train : snapshot_date <= train_end_date
        val   : train_end_date < snapshot_date <= val_end_date
        test  : val_end_date   < snapshot_date <= test_end_date

    Args:
        df       : Feature matrix with a date column and a 'churned' label
        date_col : Column containing the snapshot / event date

    Returns:
        SplitResult with .train, .val, .test DataFrames
    """
    train_end = pd.Timestamp(cfg.splits.train_end_date)
    val_end = pd.Timestamp(cfg.splits.val_end_date)
    test_end = pd.Timestamp(cfg.splits.test_end_date)

    dates = pd.to_datetime(df[date_col])

    train = df[dates <= train_end].copy()
    val = df[(dates > train_end) & (dates <= val_end)].copy()
    test = df[(dates > val_end) & (dates <= test_end)].copy()

    result = SplitResult(
        train=train,
        val=val,
        test=test,
        train_end=train_end,
        val_end=val_end,
        test_end=test_end,
    )

    logger.info(
        "Temporal split — train_end=%s  val_end=%s  test_end=%s",
        train_end.date(),
        val_end.date(),
        test_end.date(),
    )
    result.log_sizes()
    return result


def assert_no_leakage(
    split_or_train: SplitResult | pd.DataFrame,
    test: pd.DataFrame | None = None,
    date_col: str = "snapshot_date",
) -> None:
    """
    Hard assertion: no test snapshot_date appears in the train set.
    Accepts either a SplitResult or (train_df, test_df) pair.
    Raises AssertionError if leakage is detected.
    """
    if isinstance(split_or_train, SplitResult):
        train_df = split_or_train.train
        test_df = split_or_train.test
    else:
        train_df = split_or_train
        test_df = test  # type: ignore[assignment]

    max_train_date = pd.to_datetime(train_df[date_col]).max()
    min_test_date = pd.to_datetime(test_df[date_col]).min()
    assert max_train_date < min_test_date, (
        f"DATA LEAKAGE DETECTED: max train date ({max_train_date.date()}) "
        f">= min test date ({min_test_date.date()})"
    )
    logger.info(
        "Leakage check passed — max_train=%s  min_test=%s",
        max_train_date.date(),
        min_test_date.date(),
    )
