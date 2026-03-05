"""
Contract and account feature engineering.
ooc_days (days out of contract) is historically the single strongest
predictor of churn — customers near or past their contract end date
are significantly more likely to place a cease.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Mapping contract_status to an ordinal risk level
CONTRACT_STATUS_ORDINAL: dict[str, int] = {
    "01 early contract": 0,
    "02 in contract": 1,
    "03 soon to be ooc": 2,
    "04 coming ooc": 3,
    "05 newly ooc": 4,
    "06 ooc": 5,
}

# Tenure bucket boundaries in days
TENURE_BINS = [0, 90, 365, 1_095, np.inf]
TENURE_LABELS = ["0-3m", "3m-1yr", "1-3yr", "3yr+"]


def build_contract_features(
    customer_info: pd.DataFrame,
    snapshot_date: str,
) -> pd.DataFrame:
    """
    Derive contract / account risk features from a customer_info snapshot.

    Expects customer_info filtered to a single month (datevalue == snapshot month)
    or the most recent snapshot before snapshot_date per customer.

    Args:
        customer_info : Raw customer_info DataFrame (can be a full monthly table)
        snapshot_date : ISO date string — used for recency-relative features

    Returns:
        DataFrame with one row per customer and enriched feature columns
    """
    df = customer_info.copy()
    _ = snapshot_date

    # ── Contract status ordinal encoding ──────────────────────────────────────
    df["contract_status_risk"] = (
        df["contract_status"]
        .str.lower()
        .str.strip()
        .map(CONTRACT_STATUS_ORDINAL)
        .fillna(1)  # unknown → treat as in-contract (conservative)
        .astype(int)
    )

    # ── OOC-derived features ──────────────────────────────────────────────────
    df["ooc_days"] = pd.to_numeric(df["ooc_days"], errors="coerce")
    df["is_out_of_contract"] = (df["ooc_days"] >= 0).astype(int)
    df["days_to_ooc"] = (-df["ooc_days"]).clip(lower=0)  # positive = days until OOC

    # ── Speed gap (service quality signal) ───────────────────────────────────
    df["speed"] = pd.to_numeric(df["speed"], errors="coerce")
    df["line_speed"] = pd.to_numeric(df["line_speed"], errors="coerce")
    df["speed_gap"] = (df["speed"] - df["line_speed"]).clip(lower=0)
    df["speed_gap_pct"] = (df["speed_gap"] / df["speed"].replace(0, np.nan)).fillna(0)

    # ── DD cancel features ────────────────────────────────────────────────────
    df["dd_cancel_60_day"] = pd.to_numeric(df["dd_cancel_60_day"], errors="coerce").fillna(0)
    df["has_dd_cancel"] = (df["dd_cancel_60_day"] > 0).astype(int)
    df["dd_cancel_log"] = np.log1p(df["dd_cancel_60_day"])

    df["contract_dd_cancels"] = pd.to_numeric(df["contract_dd_cancels"], errors="coerce").fillna(0)

    # ── Tenure features ───────────────────────────────────────────────────────
    df["tenure_days"] = pd.to_numeric(df["tenure_days"], errors="coerce").fillna(0)
    df["tenure_log"] = np.log1p(df["tenure_days"])
    df["tenure_bucket"] = pd.cut(
        df["tenure_days"],
        bins=TENURE_BINS,
        labels=TENURE_LABELS,
        right=False,
    ).astype(str)

    # ── Technology encoding ───────────────────────────────────────────────────
    if "technology" in df.columns:
        technology_series = df["technology"]
    elif "Technology" in df.columns:
        technology_series = df["Technology"]
    else:
        technology_series = pd.Series("unknown", index=df.index)

    tech_dummies = pd.get_dummies(technology_series, prefix="tech", dtype=int)
    df = pd.concat([df, tech_dummies], axis=1)

    # ── Retain only engineered columns + identifier ──────────────────────────
    keep = [
        "unique_customer_identifier",
        "ooc_days",
        "is_out_of_contract",
        "days_to_ooc",
        "contract_status_risk",
        "speed_gap",
        "speed_gap_pct",
        "dd_cancel_60_day",
        "has_dd_cancel",
        "dd_cancel_log",
        "contract_dd_cancels",
        "tenure_days",
        "tenure_log",
        "tenure_bucket",
        "sales_channel",
        "crm_package_name",
        *[c for c in df.columns if c.startswith("tech_")],
    ]
    keep = [c for c in keep if c in df.columns]

    logger.info("  contract features: %s customers", f"{len(df):,}")
    return df[keep]
