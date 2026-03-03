"""
Visualization module for CLV Prediction and Retention Budget Optimization.

Generates publication-ready figures: ROC curves, feature importance, and
revenue simulation plots.
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for reproducibility

from pathlib import Path
from typing import Dict, Any, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


# Style settings for consistent, professional output
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
COLORS = {"lr": "#2E86AB", "rf": "#A23B72", "targeted": "#28A745", "random": "#6C757D"}


def plot_roc_curves(
    results: Dict[str, Dict[str, Any]],
    save_path: Optional[Path] = None,
    figsize: tuple = (7, 5),
    poster_mode: bool = False,
) -> None:
    """
    Plot ROC curves for Logistic Regression and Random Forest.

    Parameters
    ----------
    results : dict
        From modeling.train_and_evaluate, with keys 'lr' and 'rf'.
    save_path : Path, optional
        Where to save the figure.
    figsize : tuple
        Figure dimensions.
    """
    if poster_mode:
        figsize = (8, 6)
    fig, ax = plt.subplots(figsize=figsize)

    for key, color in [("lr", COLORS["lr"]), ("rf", COLORS["rf"])]:
        if key not in results:
            continue
        r = results[key]
        ax.plot(
            r["fpr"], r["tpr"],
            color=color,
            lw=2,
            label=f"{r['model_name']} (AUC = {r['roc_auc']:.3f})",
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    fs_label, fs_legend = (16, 14) if poster_mode else (10, 9)  # +15% for conference viewing
    ax.set_xlabel("False Positive Rate", fontsize=fs_label)
    ax.set_ylabel("True Positive Rate", fontsize=fs_label)
    ax.set_title("ROC Curves: Churn Prediction Models", fontsize=fs_label + 1)
    ax.tick_params(axis="both", labelsize=fs_label - 1)
    ax.legend(loc="lower right", fontsize=fs_legend)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(pad=0.8)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
    else:
        plt.show()


def _shorten_feature_name(name: str) -> str:
    """Shorten feature names for poster readability. Use consistent 'TotalCharges' (not both variants)."""
    replacements = [
        ("_", " "), ("No internet service", "No net"), ("Fiber optic", "Fiber"),
        ("Electronic check", "E-check"), ("two year", "2yr"), ("one year", "1yr"),
    ]
    s = name
    for old, new in replacements:
        s = s.replace(old, new)
    # Normalize Total Charges / TotalCharges to single form
    s = s.replace("Total Charges", "TotalCharges").replace("Total charges", "TotalCharges")
    return s[:25] if len(s) > 25 else s


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    save_path: Optional[Path] = None,
    figsize: tuple = (8, 6),
    title: str = "Feature Importance",
    poster_mode: bool = False,
) -> None:
    """
    Plot feature importance as horizontal bar chart.

    Parameters
    ----------
    importance_df : pd.DataFrame
        Must have columns 'feature' and 'importance'.
    top_n : int
        Number of top features to display.
    save_path : Path, optional
    figsize : tuple
    title : str
    poster_mode : bool
        If True, use fewer features, shorter labels, larger fonts for poster.
    """
    n = min(top_n, 8) if poster_mode else top_n
    plot_df = importance_df.head(n).sort_values("importance", ascending=True).copy()
    if len(plot_df) == 0:
        return

    if poster_mode:
        plot_df["feature"] = plot_df["feature"].apply(_shorten_feature_name)
        figsize = (8, 6)

    fig, ax = plt.subplots(figsize=figsize)
    bar_height = 0.5 if poster_mode else 0.6
    bars = ax.barh(plot_df["feature"], plot_df["importance"], height=bar_height, color=COLORS["rf"], alpha=0.88)
    fs = 16 if poster_mode else 10  # +15% for conference viewing
    ax.set_xlabel("Importance", fontsize=fs)
    ax.set_title(title, fontsize=fs + 1)
    ax.tick_params(axis="both", labelsize=fs - 1)
    ax.set_xlim(0, plot_df["importance"].max() * 1.15)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout(pad=0.6)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
    else:
        plt.show()


def plot_revenue_vs_budget(
    sweep_df: pd.DataFrame,
    save_path: Optional[Path] = None,
    figsize: tuple = (8, 5),
    poster_mode: bool = False,
) -> None:
    """
    Plot Revenue Saved vs Budget for targeted vs random strategies.

    Parameters
    ----------
    sweep_df : pd.DataFrame
        From simulation.budget_sweep with columns: budget, revenue_saved_targeted, revenue_saved_random.
    save_path : Path, optional
    figsize : tuple
    """
    if poster_mode:
        figsize = (9, 6)
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        sweep_df["budget"] / 1000,
        sweep_df["revenue_saved_targeted"] / 1000,
        color=COLORS["targeted"],
        lw=2,
        marker="o",
        markersize=4,
        label="Targeted (Expected Loss)",
    )
    ax.plot(
        sweep_df["budget"] / 1000,
        sweep_df["revenue_saved_random"] / 1000,
        color=COLORS["random"],
        lw=2,
        marker="s",
        markersize=4,
        label="Random",
    )

    fs = 16 if poster_mode else 10  # +15% for conference viewing
    ax.set_xlabel("Marketing Budget (thousands)", fontsize=fs)
    ax.set_ylabel("Revenue Saved (thousands)", fontsize=fs)
    ax.set_title("Revenue Saved vs. Retention Budget: Targeted vs. Random Strategy", fontsize=fs + 1)
    ax.tick_params(axis="both", labelsize=fs - 1)
    ax.legend(loc="lower right", fontsize=14 if poster_mode else 9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(pad=0.8)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
    else:
        plt.show()


def plot_sensitivity_analysis(
    sensitivity_df: pd.DataFrame,
    metric: str = "revenue_saved_targeted",
    save_path: Optional[Path] = None,
    figsize: tuple = (9, 6),
) -> None:
    """
    Plot sensitivity analysis: heatmap of metric vs churn_reduction and cost.

    Parameters
    ----------
    sensitivity_df : pd.DataFrame
        From simulation.sensitivity_analysis.
    metric : str
        One of: revenue_saved_targeted, roi_targeted, efficiency.
    save_path : Path, optional
    """
    pivot = sensitivity_df.pivot(
        index="churn_reduction_pct",
        columns="cost_per_intervention",
        values=metric,
    )
    pivot.index = [f"{x:.0%}" for x in pivot.index]

    # Scale revenue for display (show in thousands)
    if "revenue" in metric:
        plot_pivot = pivot / 1000
        cbar_label = "Revenue Saved (K$)"
        fmt = ".0f"
    elif "roi" in metric:
        plot_pivot = pivot * 100
        cbar_label = "ROI (%)"
        fmt = ".0f"
    else:
        plot_pivot = pivot
        cbar_label = "Efficiency (×)"
        fmt = ".2f"

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        plot_pivot,
        annot=True,
        fmt=fmt,
        cmap="YlGnBu",
        ax=ax,
        cbar_kws={"label": cbar_label},
    )
    ax.set_xlabel("Cost per Intervention ($)")
    ax.set_ylabel("Churn Reduction %")
    ax.set_title(f"Sensitivity Analysis: {metric.replace('_', ' ').title()}")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_budget_scaling(
    scaling_df: pd.DataFrame,
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 5),
) -> None:
    """
    Plot budget scaling: revenue saved, ROI, and efficiency vs budget.

    Parameters
    ----------
    scaling_df : pd.DataFrame
        From simulation.budget_scaling_analysis.
    save_path : Path, optional
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel 1: Revenue saved vs budget
    ax1 = axes[0]
    ax1.plot(
        scaling_df["budget"] / 1000,
        scaling_df["revenue_saved_targeted"] / 1000,
        color=COLORS["targeted"],
        lw=2,
        marker="o",
        markersize=3,
        label="Targeted",
    )
    ax1.plot(
        scaling_df["budget"] / 1000,
        scaling_df["revenue_saved_random"] / 1000,
        color=COLORS["random"],
        lw=2,
        marker="s",
        markersize=3,
        label="Random",
    )
    ax1.set_xlabel("Budget (thousands)")
    ax1.set_ylabel("Revenue Saved (thousands)")
    ax1.set_title("Revenue Saved vs. Budget")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: ROI vs budget
    ax2 = axes[1]
    ax2.plot(
        scaling_df["budget"] / 1000,
        scaling_df["roi_targeted"] * 100,
        color=COLORS["targeted"],
        lw=2,
        marker="o",
        markersize=3,
        label="Targeted",
    )
    ax2.plot(
        scaling_df["budget"] / 1000,
        scaling_df["roi_random"] * 100,
        color=COLORS["random"],
        lw=2,
        marker="s",
        markersize=3,
        label="Random",
    )
    ax2.set_xlabel("Budget (thousands)")
    ax2.set_ylabel("ROI (%)")
    ax2.set_title("ROI vs. Budget")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Targeting efficiency vs budget
    ax3 = axes[2]
    ax3.plot(
        scaling_df["budget"] / 1000,
        scaling_df["efficiency"],
        color="#6f42c1",
        lw=2,
        marker="o",
        markersize=3,
    )
    ax3.axhline(y=1, color="gray", linestyle="--", alpha=0.7)
    ax3.set_xlabel("Budget (thousands)")
    ax3.set_ylabel("Targeting Efficiency (×)")
    ax3.set_title("Targeting Efficiency vs. Budget")
    ax3.grid(True, alpha=0.3)

    fig.suptitle("Budget Scaling Analysis", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    save_path: Optional[Path] = None,
    figsize: tuple = (5, 4),
) -> None:
    """
    Plot confusion matrix with annotations.

    Parameters
    ----------
    cm : np.ndarray
        2x2 confusion matrix.
    model_name : str
    save_path : Path, optional
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Retained", "Churned"],
        yticklabels=["Retained", "Churned"],
        ax=ax,
        cbar_kws={"label": "Count"},
    )
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    ax.set_title(f"Confusion Matrix: {model_name}")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
