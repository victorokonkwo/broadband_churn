"""
Output formatter.
Transforms raw scores into a CRM-ready file with:
    - Risk tier (High / Medium / Low)
    - Top 3 churn drivers in human-readable text
    - Key risk signals for the retention agent
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from churn.config import cfg

logger = logging.getLogger(__name__)


def format_scored_output(
    customer_ids: pd.Series,
    probs: np.ndarray,
    top_drivers: list[list[str]],
    features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the operations-ready output DataFrame.

    Columns:
        unique_customer_identifier
        churn_probability    : calibrated score (0-1)
        risk_tier            : High / Medium / Low
        driver_1, driver_2, driver_3 : top SHAP-ranked features
        ooc_days             : days out of contract (key context for agent)
        contract_status_risk : ordinal risk level
        tenure_days          : customer tenure for context
    """
    high = cfg.scoring.high_risk_threshold
    med = cfg.scoring.medium_risk_threshold

    def _tier(p: float) -> str:
        if p >= high:
            return "High"
        elif p >= med:
            return "Medium"
        return "Low"

    n_drivers = cfg.scoring.top_n_drivers
    driver_cols = {f"driver_{i + 1}": [] for i in range(n_drivers)}
    for drivers in top_drivers:
        for i in range(n_drivers):
            col = f"driver_{i + 1}"
            driver_cols[col].append(drivers[i] if i < len(drivers) else "")

    scored = pd.DataFrame(
        {
            "unique_customer_identifier": customer_ids.values,
            "churn_probability": np.round(probs, 4),
            "risk_tier": [_tier(p) for p in probs],
            **driver_cols,
        }
    )

    # Add contextual columns for the retention agent
    context_cols = [
        "ooc_days",
        "contract_status_risk",
        "tenure_days",
        "loyalty_call_flag_30d",
        "speed_gap_pct",
    ]
    for col in context_cols:
        if col in features.columns:
            scored[col] = features[col].values

    # Sort by descending churn probability → top customer = highest priority
    scored = scored.sort_values("churn_probability", ascending=False).reset_index(drop=True)
    scored.insert(0, "priority_rank", range(1, len(scored) + 1))

    if len(scored) > 0 and (scored["risk_tier"] == "Low").all():
        high_n = max(1, int(np.ceil(len(scored) * 0.05)))
        med_n = max(high_n + 1, int(np.ceil(len(scored) * 0.20)))
        scored.loc[: high_n - 1, "risk_tier"] = "High"
        scored.loc[high_n : med_n - 1, "risk_tier"] = "Medium"
        logger.warning(
            "All scores fell below absolute thresholds; applied percentile fallback tiers "
            "(High top 5%%, Medium next 15%%)."
        )

    return scored
