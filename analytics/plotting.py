"""Automated plotting scripts for ablation paper graphics using Matplotlib & Pandas."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
from contracts.schemas import TrialResult
from analytics.metrics import AblationAnalyzer, summarize_results

logger = logging.getLogger("analytics.plotting")


def setup_plot_style() -> None:
    """Set clean publication-ready Matplotlib aesthetics."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "semibold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
    })


def plot_consensus_vs_density(
    df: pd.DataFrame,
    output_path: Union[str, Path],
) -> Path:
    """Plot Consensus/Convergence Rate (%) as a function of Counterfactual Density."""
    setup_plot_style()
    analyzer = AblationAnalyzer(df)
    summary = analyzer.get_density_summary()

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    densities_pct = summary["counterfactual_density"] * 100
    consensus_pct = summary["consensus_rate_pct"]

    ax.plot(
        densities_pct,
        consensus_pct,
        marker="o",
        color="#2563EB",
        linewidth=2.5,
        markersize=8,
        label="Consensus Rate (%)",
    )

    for x, y in zip(densities_pct, consensus_pct):
        ax.annotate(
            f"{y:.1f}%",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontweight="bold",
            color="#1E3A8A",
        )

    ax.set_xlabel("Counterfactual Agent Density (%)")
    ax.set_ylabel("Consensus Rate (%)")
    ax.set_title("Multi-Agent Convergence vs. Counterfactual Density")
    ax.set_xticks([0, 33, 66, 100])
    ax.set_ylim(0, 110)
    ax.grid(True, linestyle="--", alpha=0.5)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved consensus vs density plot to %s", out)
    return out


def plot_cost_vs_recovery(
    df: pd.DataFrame,
    output_path: Union[str, Path],
) -> Path:
    """Plot dual-axis comparison: Mean Token Cost vs. Deadlock Resolution Rate."""
    setup_plot_style()
    analyzer = AblationAnalyzer(df)
    summary = analyzer.get_density_summary()
    efficiencies = analyzer.compute_deadlock_resolution_efficiency()

    fig, ax1 = plt.subplots(figsize=(7.5, 4.8), dpi=300)
    densities_pct = summary["counterfactual_density"] * 100

    color1 = "#10B981"
    ax1.bar(
        densities_pct - 2,
        summary["mean_token_cost"],
        width=5,
        color=color1,
        alpha=0.85,
        label="Mean Token Cost",
    )
    ax1.set_xlabel("Counterfactual Agent Density (%)")
    ax1.set_ylabel("Total Tokens per Trial", color="#065F46")
    ax1.tick_params(axis="y", labelcolor="#065F46")
    ax1.set_xticks([0, 33, 66, 100])

    ax2 = ax1.twinx()
    color2 = "#7C3AED"
    res_rates = [efficiencies.get(d, 0.0) * 100.0 for d in summary["counterfactual_density"]]
    ax2.plot(
        densities_pct,
        res_rates,
        color=color2,
        linewidth=2.5,
        marker="s",
        markersize=7,
        label="Deadlock Resolution Efficiency (%)",
    )
    ax2.set_ylabel("Deadlock Resolution Efficiency (%)", color="#5B21B6")
    ax2.tick_params(axis="y", labelcolor="#5B21B6")
    ax2.set_ylim(0, 115)
    ax2.grid(False)

    plt.title("Token Cost vs. Deadlock Resolution Tradeoff")
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved cost vs recovery plot to %s", out)
    return out


def plot_intelligibility_progression(
    df: pd.DataFrame,
    output_path: Union[str, Path],
) -> Path:
    """Plot proportion of Strong vs. Ultra-Strong Intelligibility tiers."""
    setup_plot_style()
    analyzer = AblationAnalyzer(df)
    summary = analyzer.get_density_summary()

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    densities_pct = [f"{int(d*100)}%" for d in summary["counterfactual_density"]]

    ultra_rates = summary["ultra_strong_pct"]
    strong_rates = 100.0 - ultra_rates

    ax.bar(densities_pct, strong_rates, label="Strong Intelligibility", color="#64748B", alpha=0.8)
    ax.bar(densities_pct, ultra_rates, bottom=strong_rates, label="Ultra-Strong Intelligibility", color="#F59E0B", alpha=0.9)

    ax.set_xlabel("Counterfactual Agent Density")
    ax.set_ylabel("Proportion of Sessions (%)")
    ax.set_title("Michie-Baskar Intelligibility Progression Across Densities")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper left")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved intelligibility plot to %s", out)
    return out


def generate_all_ablation_plots(
    data: Union[pd.DataFrame, List[TrialResult]],
    output_dir: Union[str, Path] = "analytics/figures",
) -> Dict[str, Path]:
    """Generate all standard research figures and return their file paths."""
    df = summarize_results(data) if isinstance(data, list) else data
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig1 = plot_consensus_vs_density(df, out_dir / "fig1_consensus_vs_density.png")
    fig2 = plot_cost_vs_recovery(df, out_dir / "fig2_cost_vs_recovery.png")
    fig3 = plot_intelligibility_progression(df, out_dir / "fig3_intelligibility_progression.png")

    return {
        "consensus_vs_density": fig1,
        "cost_vs_recovery": fig2,
        "intelligibility_progression": fig3,
    }
