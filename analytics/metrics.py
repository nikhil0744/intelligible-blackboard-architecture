"""Ablation metrics calculations, token cost tracking, and statistical aggregators."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Union
from pathlib import Path

import pandas as pd
from contracts.schemas import TrialResult

logger = logging.getLogger("analytics.metrics")


def summarize_results(results: List[TrialResult]) -> pd.DataFrame:
    """Convert a list of TrialResult Pydantic objects into a structured pandas DataFrame."""
    records = []
    for r in results:
        records.append({
            "session_id": r.session_id,
            "benchmark_name": r.benchmark_name.value,
            "task_id": r.task_id,
            "counterfactual_density": float(r.counterfactual_density),
            "final_status": r.final_status.value,
            "consensus_reached": bool(r.consensus_reached),
            "turns_taken": int(r.turns_taken),
            "deadlock_count": int(r.deadlock_count),
            "counterfactual_interventions": int(r.counterfactual_interventions),
            "intelligibility_level": r.intelligibility.level.value,
            "is_ultra_strong": bool(r.intelligibility.is_ultra_strong),
            "total_token_cost": int(r.total_token_cost),
            "duration_seconds": float(r.duration_seconds),
        })
    return pd.DataFrame(records)


class AblationAnalyzer:
    """Statistical evaluator for the multi-agent counterfactual density ablation study."""

    def __init__(self, data: Union[pd.DataFrame, List[TrialResult]]):
        if isinstance(data, list):
            self.df = summarize_results(data)
        else:
            self.df = data.copy()

    def get_density_summary(self) -> pd.DataFrame:
        """
        Aggregate key performance indicators grouped by counterfactual density:
          - Consensus / Convergence Rate (%)
          - Mean Turns Taken
          - Mean Deadlock Count
          - Mean Counterfactual Interventions
          - Mean Token Cost
          - Ultra-Strong Intelligibility Rate (%)
        """
        if self.df.empty:
            return pd.DataFrame()

        grouped = self.df.groupby("counterfactual_density")
        summary = grouped.agg(
            total_trials=("session_id", "count"),
            consensus_rate=("consensus_reached", "mean"),
            mean_turns=("turns_taken", "mean"),
            deadlock_rate=("deadlock_count", lambda x: (x > 0).mean()),
            mean_interventions=("counterfactual_interventions", "mean"),
            mean_token_cost=("total_token_cost", "mean"),
            ultra_strong_rate=("is_ultra_strong", "mean"),
            mean_duration_sec=("duration_seconds", "mean"),
        ).reset_index()

        summary["consensus_rate_pct"] = summary["consensus_rate"] * 100.0
        summary["deadlock_rate_pct"] = summary["deadlock_rate"] * 100.0
        summary["ultra_strong_pct"] = summary["ultra_strong_rate"] * 100.0

        return summary

    def compute_deadlock_resolution_efficiency(self) -> Dict[float, float]:
        """Compute the percentage of encountered deadlocks that were resolved per density."""
        efficiencies: Dict[float, float] = {}
        for density, group in self.df.groupby("counterfactual_density"):
            deadlocked = group[group["deadlock_count"] > 0]
            if deadlocked.empty:
                efficiencies[float(density)] = 1.0
            else:
                resolved = deadlocked[deadlocked["consensus_reached"]]
                efficiencies[float(density)] = len(resolved) / len(deadlocked)
        return efficiencies

    def export_summary_table(self, output_path: Union[str, Path]) -> None:
        """Save LaTeX and Markdown tabular representations for paper inclusion."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        summary = self.get_density_summary()

        md_path = path.with_suffix(".md")
        try:
            summary.to_markdown(md_path, index=False)
        except (ImportError, Exception):
            # Fallback direct markdown table generator
            headers = [str(col) for col in summary.columns]
            lines = [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |",
            ]
            for _, row in summary.iterrows():
                formatted_vals = []
                for val in row.values:
                    if isinstance(val, float):
                        formatted_vals.append(f"{val:.2f}")
                    else:
                        formatted_vals.append(str(val))
                lines.append("| " + " | ".join(formatted_vals) + " |")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        logger.info("Saved markdown ablation summary table to %s", md_path)

        csv_path = path.with_suffix(".csv")
        summary.to_csv(csv_path, index=False)
        logger.info("Saved CSV ablation summary table to %s", csv_path)
