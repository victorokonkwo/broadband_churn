"""Unit tests for evaluation metrics."""
import numpy as np


def test_metrics_return_all_keys():
    from churn.evaluation.metrics import compute_all_metrics
    rng = np.random.RandomState(42)
    y = rng.choice([0, 1], 200, p=[0.8, 0.2])
    p = rng.uniform(0, 1, 200)
    m = compute_all_metrics(y, p)
    expected_keys = {"auc_pr", "auc_roc", "f1", "brier_score",
                     "precision_at_5pct", "lift_at_5pct",
                     "precision_at_10pct", "lift_at_10pct",
                     "precision_at_20pct", "lift_at_20pct"}
    assert expected_keys.issubset(set(m.keys()))


def test_perfect_model_auc_pr():
    from churn.evaluation.metrics import compute_all_metrics
    y = np.array([0]*80 + [1]*20)
    p = np.array([0.1]*80 + [0.9]*20)
    m = compute_all_metrics(y, p)
    assert m["auc_pr"] > 0.95


def test_decile_table_shape():
    from churn.evaluation.metrics import decile_table
    rng = np.random.RandomState(42)
    y = rng.choice([0, 1], 1000, p=[0.9, 0.1])
    p = rng.uniform(0, 1, 1000)
    dt = decile_table(y, p)
    assert len(dt) == 10
    assert "lift" in dt.columns
    assert dt["cumulative_capture_rate"].iloc[-1] > 0.95  # Should capture nearly all
