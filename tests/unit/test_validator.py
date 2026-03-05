"""Unit tests for Pandera data validation schemas."""

import pandas as pd
import pytest
from pandera.errors import SchemaError


def test_cease_schema_valid(sample_cease_df):
    from churn.data.validator import validate_cease

    result = validate_cease(sample_cease_df)
    assert len(result) == 100


def test_cease_schema_rejects_invalid_reason():
    from churn.data.validator import validate_cease

    bad = pd.DataFrame(
        {
            "unique_customer_identifier": ["cust_0"],
            "cease_placed_date": [pd.Timestamp("2023-06-01")],
            "cease_completed_date": [pd.NaT],
            "reason_description": ["Unknown"],
            "reason_description_insight": ["INVALID_CATEGORY"],
        }
    )
    with pytest.raises(SchemaError):
        validate_cease(bad)


def test_cease_schema_rejects_date_inversion():
    from churn.data.validator import validate_cease

    bad = pd.DataFrame(
        {
            "unique_customer_identifier": ["cust_0"],
            "cease_placed_date": [pd.Timestamp("2023-06-01")],
            "cease_completed_date": [pd.Timestamp("2023-05-01")],  # before placed
            "reason_description": ["Competitor"],
            "reason_description_insight": ["CompetitorDeals"],
        }
    )
    with pytest.raises(SchemaError):
        validate_cease(bad)
