"""
Batch scoring pipeline.
Scores all active customers (not in cease.csv) and outputs a ranked CSV
ready for the retention ops team.

Entry point: python -m churn.scoring.batch_scorer
         or: make score
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import shap

from churn.config import cfg
from churn.data.loader import get_connection
from churn.features.feature_store import build_feature_matrix
from churn.models.calibrator import ChurnCalibrator
from churn.models.lgbm_model import EXCLUDE_COLS, LGBMChurnModel
from churn.scoring.output_formatter import format_scored_output

logger = logging.getLogger(__name__)


def _resolve_artefact_path(
    artefacts_dir: Path,
    preferred_name: str,
    fallback_name: str,
) -> Path:
    preferred = artefacts_dir / preferred_name
    fallback = artefacts_dir / fallback_name
    if preferred.exists():
        return preferred
    return fallback


def get_active_customers(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Identify all customers who have NOT placed a cease.
    These are the customers we want to score for proactive retention.
    """
    sql = """
    SELECT DISTINCT ci.unique_customer_identifier
    FROM customer_info ci
    WHERE ci.unique_customer_identifier NOT IN (
        SELECT DISTINCT unique_customer_identifier FROM cease
    )
    """
    return con.execute(sql).df()


def score_active_customers(
    snapshot_date: str | None = None,
    model_path: Path | None = None,
    calibrator_path: Path | None = None,
) -> pd.DataFrame:
    """
    Score all active customers for churn propensity.

    Steps:
        1. Load fitted model + calibrator
        2. Build features for active customers (as of today)
        3. Score with calibrated model
        4. Enrich with SHAP top drivers per customer
        5. Format as operations-ready CSV

    Returns:
        Scored DataFrame written to outputs/scored_customers.csv
    """
    artefacts = cfg.paths.model_artefacts_dir
    model_path = model_path or _resolve_artefact_path(
        artefacts,
        preferred_name="lgbm_churn_model.pkl",
        fallback_name="lgbm_model.joblib",
    )
    calibrator_path = calibrator_path or _resolve_artefact_path(
        artefacts,
        preferred_name="calibrator.pkl",
        fallback_name="calibrator.joblib",
    )

    model = LGBMChurnModel.load(model_path)
    calibrator = ChurnCalibrator.load(calibrator_path)

    try:
        con = get_connection()
    except duckdb.IOException:
        logger.warning("DuckDB write lock detected; trying read-only connection for scoring.")
        try:
            con = get_connection(read_only=True)
        except duckdb.IOException:
            source_db = cfg.paths.duckdb_path
            snapshot_db = cfg.paths.processed_data_dir / "churn_scoring_snapshot.db"
            shutil.copy2(source_db, snapshot_db)
            logger.warning(
                "DuckDB lock persists; using snapshot copy for scoring: %s",
                snapshot_db,
            )
            con = get_connection(db_path=snapshot_db, read_only=True)

    if snapshot_date is None:
        snap = str(
            con.execute("SELECT CAST(MAX(datevalue) AS DATE) FROM customer_info").fetchone()[0]
        )
    else:
        snap = snapshot_date

    logger.info("Scoring active customers as of %s …", snap)

    # Build features
    features, _ = build_feature_matrix(snap, con, is_training=False)

    # Score
    probs = calibrator.predict_proba(features)

    # Get SHAP top drivers per customer
    logger.info("Computing per-customer SHAP explanations …")
    feature_cols = [c for c in features.columns if c not in EXCLUDE_COLS]
    X_feat = features[feature_cols].copy()
    model_feature_names = model.feature_names or list(model.booster.feature_name())
    if model_feature_names:
        missing = [c for c in model_feature_names if c not in X_feat.columns]
        for col in missing:
            X_feat[col] = 0.0
        X_feat = X_feat[model_feature_names]
    aligned_feature_names = list(X_feat.columns)

    explainer = shap.TreeExplainer(model.booster)
    shap_values = explainer(X_feat)

    top_drivers = []
    for i in range(len(features)):
        vals = shap_values.values[i]
        top3_idx = np.argsort(np.abs(vals))[::-1][: cfg.scoring.top_n_drivers]
        drivers = [aligned_feature_names[j] for j in top3_idx]
        top_drivers.append(drivers)

    # Format output
    scored = format_scored_output(
        customer_ids=features["unique_customer_identifier"],
        probs=probs,
        top_drivers=top_drivers,
        features=features,
    )

    # Save
    out_path = cfg.paths.scored_customers_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_path, index=False)
    logger.info(
        "Scored %s customers → %s  (High: %s  Medium: %s  Low: %s)",
        f"{len(scored):,}",
        out_path,
        (scored["risk_tier"] == "High").sum(),
        (scored["risk_tier"] == "Medium").sum(),
        (scored["risk_tier"] == "Low").sum(),
    )
    return scored


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    score_active_customers()
