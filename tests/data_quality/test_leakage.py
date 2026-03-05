"""
Data leakage tests — THE most important quality gate.
If any of these fail, CI must block the PR.
Leakage = using information from the future to make a prediction.
Common sources:
    - Call features computed after cease_placed_date
    - Usage data after cease date
    - Random (non-temporal) train/test split
"""
import numpy as np
import pandas as pd
import pytest

from churn.data.splitter import assert_no_leakage


def test_no_temporal_leakage_in_split():
    """Assert train max date < test min date in a temporal split."""
    train = pd.DataFrame({
        "snapshot_date": pd.date_range("2023-01-01", periods=100, freq="D"),
        "churned": np.random.choice([0, 1], 100),
    })
    test = pd.DataFrame({
        "snapshot_date": pd.date_range("2023-07-01", periods=50, freq="D"),
        "churned": np.random.choice([0, 1], 50),
    })
    # This should NOT raise
    assert_no_leakage(train, test, date_col="snapshot_date")


def test_leakage_detected_when_overlap():
    """Assert that overlapping dates trigger a leakage alert."""
    train = pd.DataFrame({
        "snapshot_date": pd.date_range("2023-01-01", periods=100, freq="D"),
        "churned": np.random.choice([0, 1], 100),
    })
    test = pd.DataFrame({
        "snapshot_date": pd.date_range("2023-03-01", periods=50, freq="D"),  # overlaps
        "churned": np.random.choice([0, 1], 50),
    })
    with pytest.raises(AssertionError, match="DATA LEAKAGE"):
        assert_no_leakage(train, test, date_col="snapshot_date")


def test_call_features_use_only_past_data():
    """
    Verify that call feature SQL uses 'event_date < snapshot_date'.
    This is a source-code inspection test — ensures the DuckDB SQL
    in call_features.py never queries future data.
    """
    import inspect
    from churn.features.call_features import build_call_features
    source = inspect.getsource(build_call_features)
    # The query must filter: event_date < DATE '{snapshot_date}'
    assert "event_date < DATE" in source, (
        "call_features.py MUST filter: event_date < snapshot_date"
    )


def test_usage_features_use_only_past_data():
    """Same check for usage features."""
    import inspect
    from churn.features.usage_features import build_usage_features
    source = inspect.getsource(build_usage_features)
    assert "calendar_date < DATE" in source, (
        "usage_features.py MUST filter: calendar_date < snapshot_date"
    )
