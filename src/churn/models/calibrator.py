"""
Probability calibration.
Raw LightGBM probability outputs are not well-calibrated —
a score of 0.8 does not necessarily mean an 80% churn probability.
Calibration is REQUIRED when scores are used in:
  - Business cost-benefit calculations (CLV × churn_prob)
  - The risk tier thresholds (High/Medium/Low)
  - Any communication to the business ("this customer has an 80% chance of leaving")

Method: Isotonic Regression (preferred when n > 1000) or Platt Scaling (sigmoid).
Validation: reliability diagram (calibration curve) before and after.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression as _SigmoidCalibrator

from churn.config import cfg

logger = logging.getLogger(__name__)


class SklearnModelWrapper(BaseEstimator, ClassifierMixin):
    """
    Wraps our BaseChurnModel to look like an sklearn estimator,
    enabling use with CalibratedClassifierCV.
    """

    def __init__(self, churn_model: "BaseChurnModel") -> None:  # noqa: F821
        self.churn_model = churn_model
        self.classes_ = np.array([0, 1])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SklearnModelWrapper":
        # Already fitted — calibrator just wraps, doesn't refit
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self.churn_model.predict_proba(X)
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class ChurnCalibrator:
    """
    Wraps a fitted BaseChurnModel and fits an isotonic regression
    calibrator on held-out calibration data (typically the val set).
    """

    def __init__(self, method: str = "isotonic") -> None:
        """
        Args:
            method : 'isotonic' (recommended for n > 1000) or 'sigmoid'
        """
        self.method = method
        self._model = None
        self._calibration_model = None

    def fit(
        self,
        model: "BaseChurnModel",  # noqa: F821
        X_cal: pd.DataFrame,
        y_cal: pd.Series,
    ) -> "ChurnCalibrator":
        """
        Fit the calibrator on held-out calibration data.

        Args:
            model : Already-trained churn model
            X_cal : Calibration feature set (val set recommended)
            y_cal : True labels for calibration set
        """
        self._model = model
        raw_probs = model.predict_proba(X_cal)

        if self.method == "isotonic":
            self._calibration_model = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip"
            )
            self._calibration_model.fit(raw_probs, y_cal)
        elif self.method == "sigmoid":
            self._calibration_model = _SigmoidCalibrator()
            self._calibration_model.fit(
                raw_probs.reshape(-1, 1), y_cal
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

        logger.info("Calibrator fitted (method=%s) on %s samples", self.method, f"{len(y_cal):,}")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated churn probabilities."""
        if self._calibration_model is None:
            raise RuntimeError("Calibrator not fitted. Call .fit() first.")
        raw_probs = self._model.predict_proba(X)
        if self.method == "isotonic":
            return self._calibration_model.transform(raw_probs)
        else:  # sigmoid
            return self._calibration_model.predict_proba(
                raw_probs.reshape(-1, 1)
            )[:, 1]

    def plot_calibration_curve(
        self,
        model: "BaseChurnModel",  # noqa: F821
        X: pd.DataFrame,
        y: pd.Series,
        save_path: Path | None = None,
        n_bins: int = 10,
    ) -> None:
        """
        Plot reliability diagram: before and after calibration.
        Well-calibrated model's curve should sit on the diagonal.
        """
        raw_probs = model.predict_proba(X)
        cal_probs = self.predict_proba(X)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, probs, title in [
            (axes[0], raw_probs, "Before Calibration"),
            (axes[1], cal_probs, "After Calibration"),
        ]:
            frac_pos, mean_pred = calibration_curve(y, probs, n_bins=n_bins, strategy="uniform")
            ax.plot(mean_pred, frac_pos, "s-", label="Model", color="steelblue")
            ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Fraction of positives")
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle("Probability Calibration — Reliability Diagram", fontweight="bold")
        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Calibration curve saved → %s", save_path)
        plt.show()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "ChurnCalibrator":
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return joblib.load(path)
