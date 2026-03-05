"""
Uplift modelling — finds PERSUADABLE churners, not just likely churners.
Standard churn models identify customers likely to leave.
Uplift models identify customers who will change their decision IF contacted.
Calling someone who was going to leave regardless wastes no money but has
zero retention ROI; calling someone who was going to stay wastes an agent slot.
The valuable segment is the "Persuadable" quadrant.

Implementation: T-Learner meta-learner using two LightGBM base estimators.
    T0: trained on control group (not contacted)
    T1: trained on treatment group (received retention call)
    Uplift score = T1.predict_proba - T0.predict_proba

Note: Requires a treatment indicator column ('contacted') in training data.
If no A/B test data is available, this module documents the framework for
future use once a randomised experiment is run.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TLearnerUpliftModel:
    """
    T-Learner meta-learner for uplift modelling.
    Two independent base models — one per treatment arm.

    Args:
        base_estimator_cls : Class of the base model (must have fit/predict_proba)
        treatment_col      : Column indicating treatment (1 = contacted, 0 = control)
        outcome_col        : Column indicating outcome (1 = retained after contact)
    """

    def __init__(
        self,
        base_estimator_cls: type | None = None,
        treatment_col: str = "contacted",
        outcome_col: str = "retained",
    ) -> None:
        if base_estimator_cls is None:
            from churn.models.lgbm_model import LGBMChurnModel
            base_estimator_cls = LGBMChurnModel

        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self._model_t0 = base_estimator_cls()  # control model
        self._model_t1 = base_estimator_cls()  # treatment model
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "TLearnerUpliftModel":
        """
        Fit separate models on control and treatment groups.
        Requires both treatment_col and outcome_col in df.
        """
        feature_cols = [
            c for c in df.columns
            if c not in {self.treatment_col, self.outcome_col,
                         "unique_customer_identifier", "snapshot_date", "churned"}
        ]

        control = df[df[self.treatment_col] == 0]
        treatment = df[df[self.treatment_col] == 1]

        logger.info(
            "Uplift T-Learner: control=%s, treatment=%s",
            f"{len(control):,}", f"{len(treatment):,}",
        )

        self._model_t0.fit(
            control[feature_cols], control[self.outcome_col],
        )
        self._model_t1.fit(
            treatment[feature_cols], treatment[self.outcome_col],
        )
        self._feature_cols = feature_cols
        self._fitted = True
        return self

    def predict_uplift(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute individual treatment effect (ITE) = P(retain|treated) - P(retain|control).
        Positive uplift = customer is persuadable.
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call .fit() first.")
        X = df[self._feature_cols]
        p_t1 = self._model_t1.predict_proba(X)
        p_t0 = self._model_t0.predict_proba(X)
        return p_t1 - p_t0

    def get_qini_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Segment customers into the four uplift quadrants:
            - Persuadables    : high churn risk, positive uplift → CALL
            - Sure Things     : high risk, negative uplift → Monitor
            - Lost Causes     : high risk, zero uplift → Lower priority
            - Do Not Disturbs : low risk, negative uplift → Do not contact
        """
        uplift = self.predict_uplift(df)
        churn_p = pd.Series(0.5, index=df.index)  # placeholder if no binary model

        result = df[["unique_customer_identifier"]].copy()
        result["uplift_score"] = uplift

        def segment(u: float) -> str:
            if u > 0.05:
                return "Persuadable"
            elif u < -0.05:
                return "Do Not Disturb"
            else:
                return "Neutral"

        result["uplift_segment"] = result["uplift_score"].map(segment)
        return result

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "TLearnerUpliftModel":
        with open(path, "rb") as f:
            return pickle.load(f)
