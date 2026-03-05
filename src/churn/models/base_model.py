"""
Abstract base model interface.
All churn model implementations must implement this contract.
Trainer and scoring modules only import BaseChurnModel — never the
concrete implementation — following the Dependency Inversion principle.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class BaseChurnModel(ABC):
    """Abstract interface for all churn prediction models."""

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "BaseChurnModel":
        """Train the model. Returns self for chaining."""
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability of churn for each row. Shape: (n,)"""
        ...

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions at the given threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)

    @abstractmethod
    def save(self, path: Path) -> None:
        """Serialise the model to disk."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseChurnModel":
        """Deserialise a model from disk."""
        ...

    @property
    def feature_names(self) -> list[str]:
        """Return the feature names used during training."""
        return getattr(self, "_feature_names", [])
