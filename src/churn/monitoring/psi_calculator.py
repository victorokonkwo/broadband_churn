"""
Population Stability Index (PSI) calculator.
Industry-standard metric for monitoring feature distribution shifts.

Interpretation:
    PSI < 0.1  : No significant change
    0.1 ≤ PSI < 0.2 : Moderate change — investigate
    PSI ≥ 0.2 : Significant change — trigger retrain alert

PSI = Σ (actual% - expected%) × ln(actual% / expected%)
where bins are equal-width over the reference distribution.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """
    Compute PSI between two 1-D distributions.

    Args:
        reference : Reference distribution (training data)
        current   : New distribution (scoring data)
        n_bins    : Number of equal-width bins
        epsilon   : Small constant to avoid log(0)

    Returns:
        PSI float value
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)

    # Remove NaN
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]

    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    # Create bins from reference distribution
    bin_edges = np.linspace(ref.min(), ref.max(), n_bins + 1)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts = np.histogram(ref, bins=bin_edges)[0].astype(float)
    cur_counts = np.histogram(cur, bins=bin_edges)[0].astype(float)

    # Convert to proportions
    ref_pct = ref_counts / ref_counts.sum() + epsilon
    cur_pct = cur_counts / cur_counts.sum() + epsilon

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def compute_psi_all_features(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    threshold: float = 0.2,
) -> dict[str, float]:
    """
    Compute PSI for all shared numeric columns.

    Returns:
        dict of {feature_name: psi_value}
        Features exceeding threshold are logged as warnings.
    """
    numeric_cols = reference.select_dtypes(include=["number"]).columns.tolist()
    shared = [
        c for c in numeric_cols if c in current.columns and c not in {"churned", "snapshot_date"}
    ]

    results = {}
    for col in shared:
        psi = compute_psi(reference[col].values, current[col].values)
        results[col] = round(psi, 4)
        if psi >= threshold:
            logger.warning("PSI ALERT: %s = %.4f (threshold=%.2f)", col, psi, threshold)

    n_alert = sum(1 for v in results.values() if v >= threshold)
    logger.info(
        "PSI computed for %s features — %s above threshold (%.2f)",
        len(results),
        n_alert,
        threshold,
    )
    return results
