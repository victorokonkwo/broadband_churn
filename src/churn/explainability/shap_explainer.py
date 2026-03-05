"""
Global SHAP explainability.
Uses TreeExplainer — exact, fast, specifically designed for tree models.
Produces:
    - Bar chart: mean |SHAP| per feature (top 20)
    - Beeswarm: direction + magnitude — shows HOW each feature drives churn
    - Dependence plots for top 3 features: non-linear effect shapes
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from churn.config import cfg
from churn.models.lgbm_model import LGBMChurnModel, EXCLUDE_COLS

logger = logging.getLogger(__name__)
FIG_DIR = cfg.paths.figures_dir


def compute_shap_values(
    model: LGBMChurnModel,
    X: pd.DataFrame,
    max_samples: int = 5_000,
) -> tuple[shap.Explanation, pd.DataFrame]:
    """
    Compute SHAP values for X using TreeExplainer.

    Args:
        model       : Fitted LGBMChurnModel
        X           : Feature DataFrame (with non-feature cols present)
        max_samples : Cap sample size for speed (full test set can be slow)

    Returns:
        (shap_values, X_features) — SHAP Explanation object and feature DataFrame
    """
    feature_cols = [c for c in X.columns if c not in EXCLUDE_COLS]
    X_feat = X[feature_cols]

    if len(X_feat) > max_samples:
        X_feat = X_feat.sample(max_samples, random_state=42)
        logger.info("SHAP: sampled %s rows for speed", max_samples)

    explainer = shap.TreeExplainer(model.booster)
    shap_values = explainer(X_feat)
    logger.info("SHAP values computed — shape: %s", shap_values.values.shape)
    return shap_values, X_feat


def plot_global_bar(
    shap_values: shap.Explanation,
    top_n: int = 20,
    save: bool = True,
) -> plt.Figure:
    """Mean |SHAP| bar chart — feature importance ranking."""
    fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.35)))
    shap.plots.bar(shap_values, max_display=top_n, show=False, ax=ax)
    ax.set_title("Global Feature Importance (mean |SHAP|)", fontweight="bold")
    plt.tight_layout()
    if save:
        _save(fig, "shap_bar.png")
    return fig


def plot_beeswarm(
    shap_values: shap.Explanation,
    top_n: int = 20,
    save: bool = True,
) -> plt.Figure:
    """
    Beeswarm plot — direction AND magnitude of each feature.
    Red = high feature value, blue = low.
    Right = pushes towards churn (positive SHAP), left = away from churn.
    """
    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.35)))
    shap.plots.beeswarm(shap_values, max_display=top_n, show=False, ax=ax)
    ax.set_title("SHAP Beeswarm — Feature Impact on Churn Probability", fontweight="bold")
    plt.tight_layout()
    if save:
        _save(fig, "shap_beeswarm.png")
    return fig


def plot_dependence(
    shap_values: shap.Explanation,
    X_feat: pd.DataFrame,
    feature: str,
    interaction_feature: str = "auto",
    save: bool = True,
) -> plt.Figure:
    """
    SHAP dependence plot for a single feature.
    Reveals non-linear effects — e.g. the cliff edge at ooc_days ≈ 0.
    """
    feat_idx = list(X_feat.columns).index(feature)
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.scatter(
        shap_values[:, feat_idx],
        color=shap_values,
        show=False,
        ax=ax,
    )
    ax.set_title(f"SHAP Dependence — {feature}", fontweight="bold")
    ax.axhline(0, color="black", linestyle="--", alpha=0.4)
    plt.tight_layout()
    if save:
        _save(fig, f"shap_dependence_{feature}.png")
    return fig


def plot_top3_dependence(
    shap_values: shap.Explanation,
    X_feat: pd.DataFrame,
    save: bool = True,
) -> list[plt.Figure]:
    """Plot dependence plots for the top 3 most important features by mean |SHAP|."""
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    top3_idx = np.argsort(mean_abs)[::-1][:3]
    top3_features = [X_feat.columns[i] for i in top3_idx]

    figs = []
    for feat in top3_features:
        fig = plot_dependence(shap_values, X_feat, feat, save=save)
        figs.append(fig)
    return figs


def _save(fig: plt.Figure, filename: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info("SHAP plot saved → %s", path)


if __name__ == "__main__":
    import pickle
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    model_path = cfg.paths.model_artefacts_dir / "lgbm_churn_model.pkl"
    model = LGBMChurnModel.load(model_path)

    test_features = sorted(cfg.paths.features_dir.glob("features_*.parquet"))
    if test_features:
        X_test = pd.read_parquet(test_features[-1])
        shap_vals, X_feat = compute_shap_values(model, X_test)
        plot_global_bar(shap_vals)
        plot_beeswarm(shap_vals)
        plot_top3_dependence(shap_vals, X_feat)
        logger.info("Global SHAP plots saved to %s", FIG_DIR)
