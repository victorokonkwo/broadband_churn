"""
Shared test fixtures.
Provides mock DataFrames for unit testing without requiring real data on disk.
"""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_cease_df() -> pd.DataFrame:
    placed_dates = pd.date_range("2023-06-01", periods=100, freq="D")
    return pd.DataFrame({
        "unique_customer_identifier": [f"cust_{i}" for i in range(100)],
        "cease_placed_date": placed_dates,
        "cease_completed_date": [
            d + pd.Timedelta(days=14) if i % 3 != 0 else pd.NaT  # 33% pending
            for i, d in enumerate(placed_dates)
        ],
        "reason_description": ["Competitor Deals - No longer required"] * 40 + ["Not Known"] * 30 + ["Home mover"] * 30,
        "reason_description_insight": ["CompetitorDeals"] * 40 + ["VagueReason"] * 30 + ["HomeMove"] * 30,
    })


@pytest.fixture
def sample_customer_info_df() -> pd.DataFrame:
    rng = np.random.RandomState(42)
    n = 200
    return pd.DataFrame({
        "unique_customer_identifier": [f"cust_{i}" for i in range(n)],
        "datevalue": pd.Timestamp("2023-05-01"),
        "contract_status": rng.choice(
            ["in contract", "out of contract", "near expiry", "early contract"], n
        ),
        "contract_dd_cancels": rng.choice([0, 0, 0, 1, 2], n).astype(float),
        "dd_cancel_60_day": rng.choice([0, 0, 0, 0, 1], n).astype(float),
        "ooc_days": rng.uniform(-200, 100, n),
        "Technology": rng.choice(["FTTC", "MPF", "G.Fast"], n),
        "speed": rng.choice([40.0, 55.0, 80.0, 160.0], n),
        "line_speed": rng.uniform(20, 80, n),
        "sales_channel": rng.choice(["Inbound", "Online- X", "Migrated Customer"], n),
        "crm_package_name": rng.choice(["Unlimited Broadband", "Essential Fibre", "Premium Fibre"], n),
        "tenure_days": rng.uniform(30, 2000, n),
    })


@pytest.fixture
def sample_feature_matrix() -> pd.DataFrame:
    """Feature matrix with a mix of churned=0/1 rows."""
    rng = np.random.RandomState(42)
    n = 500
    df = pd.DataFrame({
        "unique_customer_identifier": [f"cust_{i}" for i in range(n)],
        "snapshot_date": pd.Timestamp("2023-06-01"),
        "churned": rng.choice([0, 0, 0, 0, 1], n),  # ~20% churn rate
        "ooc_days": rng.uniform(-200, 100, n),
        "is_out_of_contract": (rng.uniform(-200, 100, n) >= 0).astype(int),
        "contract_status_risk": rng.choice([0, 1, 2, 3], n),
        "speed_gap": rng.uniform(0, 20, n),
        "speed_gap_pct": rng.uniform(0, 0.3, n),
        "dd_cancel_60_day": rng.choice([0, 0, 0, 1], n).astype(float),
        "has_dd_cancel": rng.choice([0, 0, 0, 1], n),
        "tenure_days": rng.uniform(30, 2000, n),
        "tenure_log": np.log1p(rng.uniform(30, 2000, n)),
        "call_count_30d": rng.poisson(2, n),
        "loyalty_call_flag_30d": rng.choice([0, 0, 0, 1], n),
        "avg_talk_time_30d": rng.uniform(60, 600, n),
        "avg_download_30d": rng.uniform(100, 5000, n),
        "download_trend_7_30": rng.uniform(0.5, 1.5, n),
    })
    return df
