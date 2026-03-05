"""
Feature store — single entry point for assembling the model-ready feature matrix.

Pattern mirrors production feature stores (Feast, Hopsworks):
    - Feature modules are called per-snapshot-date
    - All feature sets are merged on unique_customer_identifier
    - Output is validated against a Pandera schema
    - Materialised to data/features/ as Parquet for reproducibility
"""
from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd

from churn.config import cfg
from churn.data.loader import get_connection, load_customer_info
from churn.data.validator import validate_feature_matrix
from churn.features.call_features import build_call_features
from churn.features.contract_features import build_contract_features
from churn.features.usage_features import build_usage_features
from churn.features.target_encoder import CrossValidatedTargetEncoder

logger = logging.getLogger(__name__)

# Columns to target-encode
TARGET_ENCODE_COLS = ["crm_package_name", "sales_channel"]


def build_churn_label(
    customer_ids: pd.Series,
    snapshot_date: str,
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """
    Build binary churn label for each customer.
    A customer is labelled churned=1 if they placed a cease within
    [snapshot_date, snapshot_date + prediction_window_days].
    """
    window = cfg.label.prediction_window_days
    include_pending = cfg.label.include_pending_ceases

    completed_filter = "" if include_pending else "AND cease_completed_date IS NOT NULL"

    label_sql = f"""
    SELECT
        unique_customer_identifier,
        1 AS churned
    FROM cease
    WHERE cease_placed_date >= DATE '{snapshot_date}'
      AND cease_placed_date <  DATE '{snapshot_date}' + INTERVAL {window} DAY
      {completed_filter}
    GROUP BY unique_customer_identifier
    """
    churned_ids = con.execute(label_sql).df()

    all_customers = pd.DataFrame({"unique_customer_identifier": customer_ids.unique()})
    labels = all_customers.merge(churned_ids, on="unique_customer_identifier", how="left")
    labels["churned"] = labels["churned"].fillna(0).astype(int)
    labels["snapshot_date"] = pd.Timestamp(snapshot_date)
    return labels


def build_feature_matrix(
    snapshot_date: str,
    con: duckdb.DuckDBPyConnection | None = None,
    target_encoder: CrossValidatedTargetEncoder | None = None,
    is_training: bool = True,
) -> pd.DataFrame:
    """
    Assemble the full feature matrix for a given snapshot date.

    Steps:
        1. Load customer_info snapshot closest to snapshot_date
        2. Build contract features
        3. Build call features (DuckDB aggregation)
        4. Build usage features (DuckDB aggregation)
        5. Merge all on unique_customer_identifier
        6. Attach churn label
        7. Apply target encoding
        8. Validate output schema

    Args:
        snapshot_date    : ISO date string — the reference point for all features
        con              : DuckDB connection (created if None)
        target_encoder   : Pre-fitted encoder for val/test/scoring;
                           None = fit a new encoder (training mode)
        is_training      : If True and no encoder provided, fits a new encoder

    Returns:
        Validated feature matrix DataFrame
    """
    if con is None:
        con = get_connection()

    logger.info("=" * 60)
    logger.info("Building feature matrix — snapshot_date=%s", snapshot_date)

    # 1. Customer info — most recent snapshot <= snapshot_date per customer
    ci_sql = f"""
    SELECT ci.*
    FROM customer_info ci
    INNER JOIN (
        SELECT unique_customer_identifier, MAX(datevalue) AS latest_date
        FROM customer_info
        WHERE datevalue <= DATE '{snapshot_date}'
        GROUP BY unique_customer_identifier
    ) latest USING (unique_customer_identifier)
    WHERE ci.datevalue = latest.latest_date
    """
    customer_info = con.execute(ci_sql).df()
    logger.info("  customer_info: %s rows", f"{len(customer_info):,}")

    # 2. Contract features
    contract_df = build_contract_features(customer_info, snapshot_date)

    # 3. Call features
    call_df = build_call_features(snapshot_date, con)

    # 4. Usage features
    usage_df = build_usage_features(snapshot_date, con)

    # 5. Merge
    df = (
        contract_df
        .merge(call_df, on="unique_customer_identifier", how="left")
        .merge(usage_df, on="unique_customer_identifier", how="left")
    )
    logger.info("  merged feature set: %s rows × %s cols", *df.shape)

    # 6. Churn labels
    labels = build_churn_label(df["unique_customer_identifier"], snapshot_date, con)
    df = df.merge(labels, on="unique_customer_identifier", how="left")

    # 7. Target encoding
    if is_training and target_encoder is None:
        target_encoder = CrossValidatedTargetEncoder(columns=TARGET_ENCODE_COLS)
        df = target_encoder.fit_transform(df)
    elif target_encoder is not None:
        df = target_encoder.transform(df)

    # 8. Validate
    validate_feature_matrix(df)

    churn_rate = df["churned"].mean()
    logger.info(
        "  final matrix: %s rows × %s cols  churn_rate=%.3f",
        *df.shape, churn_rate,
    )
    return df, target_encoder


def materialise_features(
    snapshot_dates: list[str],
    output_dir: Path | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
) -> None:
    """
    Build and save feature matrices for a list of snapshot dates.
    Writes Parquet files to data/features/ for reproducibility.
    """
    output_dir = output_dir or cfg.paths.features_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if con is None:
        con = get_connection()

    encoder = None
    for i, snap in enumerate(snapshot_dates):
        is_training = i == 0  # fit encoder on first snapshot only
        df, encoder = build_feature_matrix(snap, con, encoder, is_training=is_training)
        out_path = output_dir / f"features_{snap}.parquet"
        df.to_parquet(out_path, index=False)
        logger.info("  Saved → %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    snaps = [
        cfg.splits.train_end_date,
        cfg.splits.val_end_date,
        cfg.splits.test_end_date,
    ]
    materialise_features(snaps)
    logger.info("Feature materialisation complete.")
