"""
Survival analysis models — complementary to the binary classifier.
Answers "WHEN will this customer churn?" rather than just "WILL they?".
Enables urgency-ranked retention call queues:
    high probability + imminent (low expected days) = call today.

Models:
    CoxPHChurnModel   : Semi-parametric Cox Proportional Hazards (lifelines)
    WeibullAFTChurnModel : Parametric Weibull Accelerated Failure Time (lifelines)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import pandas as pd
from lifelines import CoxPHFitter, WeibullAFTFitter

logger = logging.getLogger(__name__)

SURVIVAL_EXCLUDE = {
    "unique_customer_identifier",
    "snapshot_date",
    "churned",
    "tenure_bucket",
    "sales_channel",
    "crm_package_name",
}


class CoxPHChurnModel:
    """
    Cox Proportional Hazards model.
    duration_col = tenure_days (how long the customer has been with us)
    event_col    = churned     (1 = experienced the event = churned)
    """

    def __init__(self, penalizer: float = 0.1, l1_ratio: float = 0.0) -> None:
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self._model = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
        self.duration_col = "tenure_days"
        self.event_col = "churned"

    def _prep(self, df: pd.DataFrame) -> pd.DataFrame:
        keep = [
            c for c in df.columns if c not in SURVIVAL_EXCLUDE - {self.duration_col, self.event_col}
        ]
        return df[keep].select_dtypes(include=["number"])

    def fit(self, df: pd.DataFrame) -> CoxPHChurnModel:
        data = self._prep(df)
        self._model.fit(data, duration_col=self.duration_col, event_col=self.event_col)
        logger.info("CoxPH fitted — concordance index: %.4f", self._model.concordance_index_)
        return self

    def predict_median_survival(self, df: pd.DataFrame) -> pd.Series:
        """Return estimated median remaining lifetime (days) for each customer."""
        data = self._prep(df)
        sf = self._model.predict_survival_function(data)
        # Median = time at which survival function crosses 0.5
        medians = sf.apply(
            lambda col: col[col <= 0.5].index[0] if (col <= 0.5).any() else sf.index[-1]
        )
        return pd.Series(medians.values, index=df.index, name="expected_days_to_churn")

    def predict_churn_probability(self, df: pd.DataFrame, at_days: int = 90) -> pd.Series:
        """Probability of churning within at_days from today."""
        data = self._prep(df)
        sf = self._model.predict_survival_function(data, times=[at_days])
        return pd.Series(1 - sf.iloc[0].values, index=df.index, name="churn_prob_survival")

    def print_summary(self) -> None:
        self._model.print_summary()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> CoxPHChurnModel:
        with path.open("rb") as f:
            return pickle.load(f)


class WeibullAFTChurnModel:
    """
    Parametric Weibull Accelerated Failure Time model.
    More interpretable than Cox PH — directly models the time scale.
    Useful when you need to explain to business how features shift expected churn timing.
    """

    def __init__(self, penalizer: float = 0.0) -> None:
        self.penalizer = penalizer
        self._model = WeibullAFTFitter(penalizer=penalizer)
        self.duration_col = "tenure_days"
        self.event_col = "churned"

    def _prep(self, df: pd.DataFrame) -> pd.DataFrame:
        keep = [
            c for c in df.columns if c not in SURVIVAL_EXCLUDE - {self.duration_col, self.event_col}
        ]
        return df[keep].select_dtypes(include=["number"])

    def fit(self, df: pd.DataFrame) -> WeibullAFTChurnModel:
        data = self._prep(df)
        self._model.fit(data, duration_col=self.duration_col, event_col=self.event_col)
        logger.info("WeibullAFT fitted")
        return self

    def predict_median_survival(self, df: pd.DataFrame) -> pd.Series:
        data = self._prep(df)
        medians = self._model.predict_median(data)
        return pd.Series(medians.values, index=df.index, name="expected_days_to_churn")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> WeibullAFTChurnModel:
        with path.open("rb") as f:
            return pickle.load(f)
