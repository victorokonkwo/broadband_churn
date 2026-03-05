"""
Evaluation metrics suite.
Primary metric: AUC-PR (Average Precision) — correct for class imbalance.
AUC-ROC is included but is misleadingly optimistic on imbalanced datasets.
Business-facing metrics: Lift@K and Precision@K.
"""
from __future__ import annotations

import logging

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_all_metrics(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute the full evaluation metric suite.

    Returns:
        dict with keys: auc_pr, auc_roc, f1, brier_score,
                        precision_at_10, lift_at_10,
                        precision_at_20, lift_at_20
    """
    y = np.asarray(y_true)
    p = np.asarray(y_prob)
    y_pred = (p >= threshold).astype(int)

    metrics: dict[str, float] = {}

    metrics["auc_pr"]     = average_precision_score(y, p)
    metrics["auc_roc"]    = roc_auc_score(y, p)
    metrics["f1"]         = f1_score(y, y_pred, zero_division=0)
    metrics["brier_score"] = brier_score_loss(y, p)

    # Precision@K and Lift@K (K = % of customers contacted)
    base_rate = y.mean()
    for k_pct in [5, 10, 20]:
        k = max(1, int(len(y) * k_pct / 100))
        top_k_idx = np.argsort(p)[::-1][:k]
        prec_k = y[top_k_idx].mean()
        lift_k = prec_k / base_rate if base_rate > 0 else 0.0
        metrics[f"precision_at_{k_pct}pct"] = prec_k
        metrics[f"lift_at_{k_pct}pct"]      = lift_k

    logger.info(
        "Metrics — AUC-PR=%.4f  AUC-ROC=%.4f  F1=%.4f  "
        "Lift@10%%=%.2f  Precision@10%%=%.4f",
        metrics["auc_pr"], metrics["auc_roc"], metrics["f1"],
        metrics["lift_at_10pct"], metrics["precision_at_10pct"],
    )
    return metrics


def log_metrics_to_mlflow(metrics: dict[str, float]) -> None:
    """Log all metrics to the active MLflow run."""
    mlflow.log_metrics({k: round(float(v), 6) for k, v in metrics.items()})


def decile_table(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
) -> pd.DataFrame:
    """
    Build a decile-level lift table — the key business evaluation tool.
    Shows what % of true churners are captured in each score decile.

    Returns DataFrame with columns:
        decile, n_customers, n_churners, churn_rate,
        lift, cumulative_capture_rate
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob)})
    df = df.sort_values("p", ascending=False).reset_index(drop=True)
    df["decile"] = pd.qcut(df.index, q=10, labels=range(1, 11))

    base_rate = df["y"].mean()
    total_churners = df["y"].sum()

    rows = []
    cum_captured = 0
    for dec in range(1, 11):
        mask = df["decile"] == dec
        n = mask.sum()
        pos = df.loc[mask, "y"].sum()
        cum_captured += pos
        rate = pos / n if n > 0 else 0.0
        rows.append({
            "decile": dec,
            "n_customers": n,
            "n_churners": int(pos),
            "churn_rate": round(rate, 4),
            "lift": round(rate / base_rate, 2) if base_rate > 0 else 0.0,
            "cumulative_capture_rate": round(cum_captured / total_churners, 4) if total_churners > 0 else 0.0,
        })

    return pd.DataFrame(rows)
