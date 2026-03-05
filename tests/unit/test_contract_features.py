"""Unit tests for contract feature engineering."""


def test_speed_gap_computed(sample_customer_info_df):
    from churn.features.contract_features import build_contract_features

    result = build_contract_features(sample_customer_info_df, "2023-06-01")
    assert "speed_gap" in result.columns
    assert "speed_gap_pct" in result.columns
    assert (result["speed_gap"] >= 0).all()


def test_ooc_binary_flag(sample_customer_info_df):
    from churn.features.contract_features import build_contract_features

    result = build_contract_features(sample_customer_info_df, "2023-06-01")
    assert set(result["is_out_of_contract"].unique()).issubset({0, 1})


def test_tenure_bucket_labels(sample_customer_info_df):
    from churn.features.contract_features import build_contract_features

    result = build_contract_features(sample_customer_info_df, "2023-06-01")
    assert "tenure_bucket" in result.columns
    valid = {"0-3m", "3m-1yr", "1-3yr", "3yr+"}
    assert set(result["tenure_bucket"].unique()).issubset(valid)


def test_tech_dummies_created(sample_customer_info_df):
    from churn.features.contract_features import build_contract_features

    result = build_contract_features(sample_customer_info_df, "2023-06-01")
    tech_cols = [c for c in result.columns if c.startswith("tech_")]
    assert len(tech_cols) > 0
