"""Unit tests for the temporal splitter."""

import pandas as pd
import pytest


def test_temporal_split_no_overlap(sample_feature_matrix):
    from churn.data.splitter import temporal_split

    # Assign varying dates
    dates = pd.date_range("2023-01-01", periods=len(sample_feature_matrix), freq="D")
    df = sample_feature_matrix.copy()
    df["snapshot_date"] = dates[: len(df)]

    split = temporal_split(df, date_col="snapshot_date")

    # Assert no date overlap between train and test
    max_train = pd.to_datetime(split.train["snapshot_date"]).max()
    min_test = pd.to_datetime(split.test["snapshot_date"]).min()
    assert max_train < min_test, "Train dates must be strictly before test dates"


def test_assert_no_leakage_raises_on_leak(sample_feature_matrix):
    from churn.data.splitter import assert_no_leakage

    df = sample_feature_matrix.copy()
    # Both train and test have same date → should raise
    df["snapshot_date"] = pd.Timestamp("2023-07-01")
    with pytest.raises(AssertionError, match="DATA LEAKAGE DETECTED"):
        assert_no_leakage(df, df, date_col="snapshot_date")
