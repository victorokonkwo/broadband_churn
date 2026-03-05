"""
Evaluation plots.
All plots saved to outputs/figures/ and returned as matplotlib Figures.
"""

from __future__ import annotations

import logging
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import precision_recall_curve, roc_curve

from churn.config import cfg
from churn.evaluation.metrics import decile_table

logger = logging.getLogger(__name__)

FIG_DIR = cfg.paths.figures_dir


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    auc_pr: float | None = None,
    ax: plt.Axes | None = None,
    save: bool = True,
) -> plt.Figure:
    """PR curve with iso-F1 contours and the operational threshold point."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    base_rate = y_true.mean()

    fig, ax = plt.subplots(figsize=(8, 6))

    # Iso-F1 contours
    f_scores = np.linspace(0.1, 0.9, 9)
    for f_score in f_scores:
        x = np.linspace(0.01, 1.0, 100)
        y_iso = f_score * x / (2 * x - f_score)
        ax.plot(x[y_iso >= 0], y_iso[y_iso >= 0], "k--", alpha=0.1, linewidth=0.8)
        ax.annotate(
            f"F1={f_score:.1f}", xy=(0.9, y_iso[np.argmin(abs(x - 0.9))]), fontsize=7, color="grey"
        )

    ax.plot(recall, precision, "steelblue", linewidth=2, label=f"LightGBM (AUC-PR={auc_pr:.3f})")
    ax.axhline(base_rate, color="red", linestyle="--", label=f"No-skill baseline ({base_rate:.3f})")

    ax.set_xlabel("Recall (% of churners captured)")
    ax.set_ylabel("Precision (% of contacts who churn)")
    ax.set_title("Precision–Recall Curve", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        _save(fig, "pr_curve.png")
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    auc_roc: float | None = None,
    ax: plt.Axes | None = None,
    save: bool = True,
) -> plt.Figure:
    from sklearn.metrics import roc_auc_score

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    if auc_roc is None:
        auc_roc = roc_auc_score(y_true, y_prob)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig_obj = ax.get_figure()
        if fig_obj is None:
            fig_obj = plt.figure(figsize=(7, 6))
        fig = cast(Figure, fig_obj)
    ax.plot(fpr, tpr, "steelblue", linewidth=2, label=f"LightGBM (AUC={auc_roc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save:
        _save(fig, "roc_curve.png")
    return fig


def plot_lift_chart(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save: bool = True,
) -> plt.Figure:
    """
    Cumulative gains chart — the business-facing evaluation plot.
    X-axis: % of customer base contacted.
    Y-axis: % of total churners captured.
    'Calling the top 20% captures X% of all churners' is the key message.
    """
    df = pd.DataFrame({"y": y_true, "p": y_prob}).sort_values("p", ascending=False)
    n = len(df)
    total_pos = df["y"].sum()

    x = np.arange(1, n + 1) / n * 100
    cumulative_gains = df["y"].cumsum() / total_pos * 100
    random_line = x  # diagonal

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Cumulative gains
    axes[0].plot(x, cumulative_gains, "steelblue", linewidth=2, label="Model")
    axes[0].plot(x, random_line, "k--", label="Random baseline")
    axes[0].fill_between(x, cumulative_gains, random_line, alpha=0.1, color="steelblue")
    axes[0].axvline(20, color="red", linestyle=":", alpha=0.7, label="Top 20% contacted")
    axes[0].set_xlabel("% of customers contacted (by descending churn score)")
    axes[0].set_ylabel("% of churners captured")
    axes[0].set_title("Cumulative Gains Chart", fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Lift per decile
    dec_table = decile_table(y_true, y_prob)
    colors = [
        "#d73027" if lift_value >= 2 else "#4393c3" if lift_value >= 1 else "#e0e0e0"
        for lift_value in dec_table["lift"]
    ]
    axes[1].bar(dec_table["decile"], dec_table["lift"], color=colors, edgecolor="white")
    axes[1].axhline(1.0, color="black", linestyle="--", alpha=0.5, label="No lift baseline")
    axes[1].set_xlabel("Decile (1 = highest churn score)")
    axes[1].set_ylabel("Lift vs. random")
    axes[1].set_title("Lift by Decile", fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.suptitle("Business Evaluation — Churn Model Performance", fontweight="bold", y=1.02)
    plt.tight_layout()

    if save:
        _save(fig, "lift_chart.png")
    return fig


def plot_score_distribution(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ax: plt.Axes | None = None,
    save: bool = True,
) -> plt.Figure:
    """Distribution of churn scores by true label \u2014 visual separation check."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig_obj = ax.get_figure()
        if fig_obj is None:
            fig_obj = plt.figure(figsize=(8, 5))
        fig = cast(Figure, fig_obj)
    bins = np.linspace(0, 1, 40).tolist()
    ax.hist(
        y_prob[y_true == 0], bins=bins, alpha=0.6, label="Retained", color="steelblue", density=True
    )
    ax.hist(
        y_prob[y_true == 1], bins=bins, alpha=0.6, label="Churned", color="tomato", density=True
    )
    ax.set_xlabel("Churn probability score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution by True Label", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save:
        _save(fig, "score_distribution.png")
    return fig


# Aliases for notebook imports
plot_pr_curve = plot_precision_recall_curve


def _save(fig: plt.Figure, filename: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info("Plot saved → %s", path)
