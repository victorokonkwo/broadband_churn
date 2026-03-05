"""
Churn segment clustering using SHAP values.
Instead of clustering on raw features (which mixes uninformative features),
we cluster on SHAP values — each customer's position in SHAP space reflects
WHAT is driving their churn risk, not just correlated feature values.

Output segments map directly to retention actions:
    Segment A — Contract Expiry: ooc_days dominant → proactive renewal call
    Segment B — Service Issue  : speed_gap + usage_drop dominant → engineer visit
    Segment C — Price Sensitive: loyalty_call + CompetitorDeals signals → counter-offer
    Segment D — Financial Risk : dd_cancel signals dominant → payment plan discussion
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from churn.config import cfg

logger = logging.getLogger(__name__)
FIG_DIR = cfg.paths.figures_dir

SEGMENT_LABELS = {
    0: "Contract Expiry Risk",
    1: "Service Quality Issue",
    2: "Price Sensitive",
    3: "Financial Risk",
}


def cluster_on_shap_values(
    shap_values: shap.Explanation,
    X_feat: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    K-Means clustering on SHAP values (not raw features).

    Args:
        shap_values : SHAP Explanation from TreeExplainer
        X_feat      : Feature DataFrame (same rows as shap_values)
        n_clusters  : Number of customer segments

    Returns:
        DataFrame with columns: unique_customer_identifier (if present),
        cluster_id, segment_label, plus SHAP values for top features.
    """
    shap_matrix = shap_values.values

    # Standardise before clustering
    scaler = StandardScaler()
    shap_scaled = scaler.fit_transform(shap_matrix)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_ids = km.fit_predict(shap_scaled)

    result = X_feat.copy()
    result["cluster_id"] = cluster_ids
    result["segment_label"] = result["cluster_id"].map(SEGMENT_LABELS).fillna("Other")

    # Add top SHAP driver per cluster
    feature_names = list(X_feat.columns)
    result["shap_top_driver"] = [
        feature_names[np.argmax(np.abs(shap_values.values[i]))] for i in range(len(result))
    ]

    _log_cluster_summary(result, shap_values, feature_names)
    return result


def segment_by_shap(
    shap_values: shap.Explanation,
    X_feat: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Alias that returns (labels_array, full_result_df) for notebook compat."""
    result = cluster_on_shap_values(shap_values, X_feat, n_clusters, random_state)
    return result["cluster_id"].values, result


def _log_cluster_summary(
    df: pd.DataFrame,
    shap_values: shap.Explanation,
    feature_names: list[str],
) -> None:
    for cid in sorted(df["cluster_id"].unique()):
        mask = df["cluster_id"] == cid
        n = mask.sum()
        cluster_shap = shap_values.values[mask]
        top_feat = feature_names[np.argmax(np.abs(cluster_shap).mean(axis=0))]
        label = SEGMENT_LABELS.get(cid, "?")
        logger.info("  Cluster %s (%s): n=%s  top_driver=%s", cid, label, f"{n:,}", top_feat)


def plot_segment_profiles(
    df: pd.DataFrame,
    shap_values: shap.Explanation,
    feature_names: list[str],
    top_n_features: int = 8,
    save: bool = True,
) -> plt.Figure:
    """
    Heatmap of mean SHAP value per cluster per top feature.
    Helps name each cluster based on which features dominate.
    """
    top_feat_idx = np.argsort(np.abs(shap_values.values).mean(axis=0))[::-1][:top_n_features]
    top_feat_names = [feature_names[i] for i in top_feat_idx]

    n_clusters = df["cluster_id"].nunique()
    heatmap_data = np.zeros((n_clusters, top_n_features))

    for cid in range(n_clusters):
        mask = (df["cluster_id"] == cid).values
        heatmap_data[cid] = shap_values.values[mask][:, top_feat_idx].mean(axis=0)

    fig, ax = plt.subplots(figsize=(12, max(4, n_clusters * 1.2)))
    im = ax.imshow(
        heatmap_data,
        cmap="RdBu_r",
        aspect="auto",
        vmin=-np.max(np.abs(heatmap_data)),
        vmax=np.max(np.abs(heatmap_data)),
    )
    plt.colorbar(im, ax=ax, label="Mean SHAP value")

    cluster_labels = [f"Cluster {c}: {SEGMENT_LABELS.get(c, 'Unknown')}" for c in range(n_clusters)]
    ax.set_yticks(range(n_clusters))
    ax.set_yticklabels(cluster_labels)
    ax.set_xticks(range(top_n_features))
    ax.set_xticklabels(top_feat_names, rotation=45, ha="right")
    ax.set_title("Churn Segment Profiles — Mean SHAP Value per Cluster", fontweight="bold")
    plt.tight_layout()

    if save:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        path = FIG_DIR / "shap_segment_profiles.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Segment profile plot saved → %s", path)

    return fig
