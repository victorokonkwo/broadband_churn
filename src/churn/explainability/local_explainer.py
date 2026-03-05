"""
Local (per-customer) SHAP explainability.
Answers: "WHY is THIS customer predicted to churn?"
Waterfall plots show the exact contribution of each feature for one customer.
These are invaluable in the retention team's UI — agents see WHY a customer
is flagged so they can tailor the retention conversation.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from churn.config import cfg
from churn.models.lgbm_model import LGBMChurnModel, EXCLUDE_COLS

logger = logging.getLogger(__name__)
FIG_DIR = cfg.paths.figures_dir


def get_top_drivers(
    shap_values: shap.Explanation,
    row_idx: int,
    feature_names: list[str],
    n: int = 3,
) -> list[dict]:
    """
    Extract top N churn driver features for a single customer.

    Returns:
        List of dicts: [{feature, shap_value, feature_value}, ...]
        Sorted by |SHAP| descending.
    """
    vals = shap_values.values[row_idx]
    data = shap_values.data[row_idx]

    sorted_idx = np.argsort(np.abs(vals))[::-1][:n]
    return [
        {
            "feature": feature_names[i],
            "shap_value": round(float(vals[i]), 4),
            "feature_value": data[i],
        }
        for i in sorted_idx
    ]


def plot_waterfall(
    shap_values: shap.Explanation,
    row_idx: int,
    customer_id: str | None = None,
    churn_prob: float | None = None,
    save: bool = True,
    filename: str | None = None,
) -> plt.Figure:
    """
    Waterfall plot for a single customer.
    Shows base value + each feature's contribution to the final score.

    Args:
        shap_values  : SHAP Explanation object (full test set)
        row_idx      : Index of the customer in the explanation
        customer_id  : Optional — displayed in title
        churn_prob   : Optional — calibrated probability for subtitle
        save         : Whether to save to outputs/figures/
        filename     : Override output filename
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values[row_idx], show=False, ax=ax)

    title_parts = ["SHAP Waterfall — Individual Customer Explanation"]
    if customer_id:
        title_parts.append(f"Customer: {customer_id[:12]}…")
    if churn_prob is not None:
        title_parts.append(f"Churn probability: {churn_prob:.1%}")

    ax.set_title("\n".join(title_parts), fontweight="bold")
    plt.tight_layout()

    if save:
        fname = filename or f"shap_waterfall_customer_{row_idx}.png"
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        path = FIG_DIR / fname
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Waterfall plot saved → %s", path)

    return fig


def explain_top_customers(
    model: LGBMChurnModel,
    X: pd.DataFrame,
    y_prob: np.ndarray,
    n_customers: int = 5,
    save: bool = True,
) -> list[dict]:
    """
    Generate waterfall explanations for the top n_customers by churn score.
    Returns a list of per-customer driver dicts for use in scored output.
    """
    feature_cols = [c for c in X.columns if c not in EXCLUDE_COLS]
    X_feat = X[feature_cols]

    explainer   = shap.TreeExplainer(model.booster)
    shap_values = explainer(X_feat)

    # Get top-scored customers
    top_idx = np.argsort(y_prob)[::-1][:n_customers]

    explanations = []
    for rank, idx in enumerate(top_idx):
        cid = X.get("unique_customer_identifier", pd.Series()).iloc[idx] if "unique_customer_identifier" in X.columns else str(idx)
        prob = y_prob[idx]

        plot_waterfall(
            shap_values, idx,
            customer_id=str(cid),
            churn_prob=float(prob),
            save=save,
            filename=f"shap_waterfall_rank{rank+1}.png",
        )

        drivers = get_top_drivers(shap_values, idx, feature_cols, n=cfg.scoring.top_n_drivers)
        explanations.append({
            "rank": rank + 1,
            "customer_id": cid,
            "churn_probability": round(float(prob), 4),
            "drivers": drivers,
        })

    return explanations
