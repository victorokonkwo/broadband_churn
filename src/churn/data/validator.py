"""
Schema validation layer using Pandera.
Every DataFrame that enters the pipeline is validated before use.
A schema violation raises a SchemaError — this is a hard stop, not a warning.
"""
from __future__ import annotations

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check


# ─── Cease schema ─────────────────────────────────────────────────────────────

VALID_CHURN_REASONS = [
    "CompetitorDeals",
    "VagueReason",
    "HomeMove",
    "Bereavement",
    "Other",
    "BadDebtDisconnect",
    "CustomerServices",
    "Installation&Provisioning",
    "Mis-sold",
    "TV",
    "TechnicalIssue",
]

cease_schema = DataFrameSchema(
    columns={
        "unique_customer_identifier": Column(str, nullable=False),
        "cease_placed_date": Column("datetime64[ns]", nullable=False),
        "cease_completed_date": Column(str, nullable=True),
        "reason_description": Column(str, nullable=True),
        "reason_description_insight": Column(
            str,
            checks=Check.isin(VALID_CHURN_REASONS),
            nullable=True,
        ),
    },
    checks=[
        # cease_completed_date must be >= cease_placed_date when not null
        Check(
            lambda df: (
                df["cease_completed_date"].isna()
                | (df["cease_completed_date"] == "")
                | (pd.to_datetime(df["cease_completed_date"], errors="coerce")
                   >= pd.to_datetime(df["cease_placed_date"], errors="coerce"))
            ).all(),
            error="cease_completed_date must be >= cease_placed_date",
        )
    ],
    coerce=True,
)


# ─── Customer info schema ─────────────────────────────────────────────────────

VALID_CONTRACT_STATUSES = [
    "01 Early Contract",
    "02 In Contract",
    "03 Soon to be OOC",
    "04 Coming OOC",
    "05 Newly OOC",
    "06 OOC",
]

VALID_TECHNOLOGIES = ["FTTC", "FTTP", "GFAST", "MPF"]

customer_info_schema = DataFrameSchema(
    columns={
        "unique_customer_identifier": Column(str, nullable=False),
        "datevalue": Column("datetime64[ns]", nullable=False),
        "contract_status": Column(
            str,
            checks=Check.isin(VALID_CONTRACT_STATUSES),
            nullable=True,
        ),
        "contract_dd_cancels": Column(float, checks=Check.ge(0), nullable=True),
        "dd_cancel_60_day": Column(float, checks=Check.ge(0), nullable=True),
        "ooc_days": Column(float, nullable=True),
        "technology": Column(str, checks=Check.isin(VALID_TECHNOLOGIES), nullable=True),
        "speed": Column(float, checks=Check.ge(0), nullable=True),
        "line_speed": Column(float, checks=Check.ge(0), nullable=True),
        "sales_channel": Column(str, nullable=True),
        "crm_package_name": Column(str, nullable=True),
        "tenure_days": Column(float, checks=Check.ge(0), nullable=True),
    },
    coerce=True,
)


# ─── Calls schema ─────────────────────────────────────────────────────────────

calls_schema = DataFrameSchema(
    columns={
        "unique_customer_identifier": Column(str, nullable=False),
        "event_date": Column("datetime64[ns]", nullable=False),
        "call_type": Column(str, nullable=True),
        "talk_time_seconds": Column(float, checks=Check.ge(0), nullable=True),
        "hold_time_seconds": Column(float, checks=Check.ge(0), nullable=True),
    },
    coerce=True,
)


# ─── Usage schema ─────────────────────────────────────────────────────────────

usage_schema = DataFrameSchema(
    columns={
        "unique_customer_identifier": Column(str, nullable=False),
        "calendar_date": Column("datetime64[ns]", nullable=False),
        "usage_download_mbs": Column(str, nullable=True),   # stored as text
        "usage_upload_mbs": Column(str, nullable=True),      # stored as text
    },
    coerce=True,
)


# ─── Feature matrix schema ────────────────────────────────────────────────────

feature_matrix_schema = DataFrameSchema(
    columns={
        "unique_customer_identifier": Column(str, nullable=False),
        "snapshot_date": Column("datetime64[ns]", nullable=False),
        "churned": Column(int, checks=Check.isin([0, 1]), nullable=False),
    },
    coerce=True,
)


def validate_cease(df: "pd.DataFrame") -> "pd.DataFrame":  # noqa: F821
    return cease_schema.validate(df)


def validate_customer_info(df: "pd.DataFrame") -> "pd.DataFrame":  # noqa: F821
    return customer_info_schema.validate(df)


def validate_calls(df: "pd.DataFrame") -> "pd.DataFrame":  # noqa: F821
    return calls_schema.validate(df)


def validate_usage(df: "pd.DataFrame") -> "pd.DataFrame":  # noqa: F821
    return usage_schema.validate(df)


def validate_feature_matrix(df: "pd.DataFrame") -> "pd.DataFrame":  # noqa: F821
    return feature_matrix_schema.validate(df)
