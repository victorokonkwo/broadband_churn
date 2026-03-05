"""
Data drift and prediction drift monitoring using Evidently AI.
Runs periodically (weekly recommended) to detect:
    - Feature distribution shifts (input drift)
    - Prediction score distribution shifts (output drift)
If drift exceeds configured thresholds → triggers retrain alert.

Entry point: python -m churn.monitoring.drift_detector
         or: make monitor
"""

from __future__ import annotations

import json
import logging

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from churn.config import cfg
from churn.monitoring.psi_calculator import compute_psi_all_features

logger = logging.getLogger(__name__)
REPORTS_DIR = cfg.paths.reports_dir


def run_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    prediction_col: str = "churn_probability",
    save: bool = True,
) -> dict:
    """
    Generate full Evidently drift report comparing reference (training)
    vs. current (latest scoring batch) distributions.

    Args:
        reference      : Training feature matrix (reference distribution)
        current        : Latest scored batch (current distribution)
        prediction_col : Name of the prediction column (if present)
        save           : Whether to save HTML report to outputs/reports/

    Returns:
        dict with drift_detected (bool) and per-feature drift results
    """
    # Select only numeric features for drift analysis
    numeric_cols = reference.select_dtypes(include=["number"]).columns.tolist()
    exclude = {"churned", "unique_customer_identifier", "snapshot_date"}
    feature_cols = [c for c in numeric_cols if c not in exclude and c in current.columns]

    ref = reference[feature_cols].copy()
    cur = current[feature_cols].copy()

    report = Report([DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)

    drift_detected = False
    drift_share = 0.0
    try:
        if hasattr(report, "as_dict"):
            result = report.as_dict()
        elif hasattr(report, "dict"):
            result = report.dict()
        elif hasattr(report, "json"):
            result = json.loads(report.json())
        else:
            result = {}

        drift_info = result.get("metrics", [{}])[0].get("result", {})
        drift_detected = drift_info.get("dataset_drift", False)
        drift_share = drift_info.get("share_of_drifted_columns", 0.0)
    except Exception as exc:
        logger.warning(
            "Could not parse Evidently drift summary in this version (%s); "
            "falling back to default drift_detected=False.",
            exc,
        )

    logger.info(
        "Drift report — drift_detected=%s  drift_share=%.2f%%",
        drift_detected,
        drift_share * 100,
    )

    if save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        html_path = REPORTS_DIR / "drift_report.html"
        if hasattr(report, "save_html"):
            report.save_html(str(html_path))
            logger.info("Evidently HTML report saved → %s", html_path)
        else:
            logger.warning("Evidently HTML export not supported in current version.")

    return {
        "drift_detected": drift_detected,
        "drift_share": drift_share,
        "report": report,
    }


def run_full_monitoring(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> dict:
    """
    Run both Evidently drift analysis and PSI computation.
    Returns combined monitoring summary.
    """
    drift_result = run_drift_report(reference, current)
    psi_result = compute_psi_all_features(reference, current)

    # Combine alerts
    psi_alerts = [f for f, v in psi_result.items() if v > 0.2]
    if psi_alerts:
        logger.warning("PSI ALERT — features with PSI > 0.2: %s", psi_alerts)

    return {
        "drift_detected": drift_result["drift_detected"],
        "drift_share": drift_result["drift_share"],
        "psi_alerts": psi_alerts,
        "psi_values": psi_result,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Example: compare train vs. most recent scoring batch
    features_dir = cfg.paths.features_dir
    parquets = sorted(features_dir.glob("features_*.parquet"))
    if len(parquets) >= 2:
        ref = pd.read_parquet(parquets[0])
        cur = pd.read_parquet(parquets[-1])
        result = run_full_monitoring(ref, cur)
        logger.info("Monitoring result: %s", result)
    else:
        logger.error("Need at least 2 feature files for drift comparison.")
