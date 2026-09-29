"""Tests for Student 4 ablation metrics and plotting utilities."""

from pathlib import Path
import pytest

from analytics.metrics import AblationAnalyzer, summarize_results
from analytics.plotting import generate_all_ablation_plots
from benchmarks.mscore import BatchTrialRunner, MSCoReIngestor


@pytest.mark.asyncio
async def test_ablation_metrics_aggregation(tmp_path: Path):
    items = MSCoReIngestor.get_sample_dataset()[:2]
    runner = BatchTrialRunner(streaming_enabled=False)
    configs = runner.generate_ablation_matrix(items, densities=[0.0, 0.33, 0.66, 1.0])
    results = await runner.run_batch(configs)

    df = summarize_results(results)
    assert len(df) == 8
    assert "counterfactual_density" in df.columns
    assert "consensus_reached" in df.columns

    analyzer = AblationAnalyzer(df)
    summary = analyzer.get_density_summary()
    assert len(summary) == 4
    assert "consensus_rate_pct" in summary.columns
    assert "mean_token_cost" in summary.columns

    efficiencies = analyzer.compute_deadlock_resolution_efficiency()
    assert len(efficiencies) == 4

    # Test summary export to markdown and csv
    out_table = tmp_path / "ablation_summary"
    analyzer.export_summary_table(out_table)
    assert (tmp_path / "ablation_summary.md").exists()
    assert (tmp_path / "ablation_summary.csv").exists()


@pytest.mark.asyncio
async def test_plot_generation(tmp_path: Path):
    items = MSCoReIngestor.get_sample_dataset()[:2]
    runner = BatchTrialRunner(streaming_enabled=False)
    configs = runner.generate_ablation_matrix(items, densities=[0.0, 0.33, 0.66, 1.0])
    results = await runner.run_batch(configs)

    figures = generate_all_ablation_plots(results, output_dir=tmp_path / "figures")
    assert (tmp_path / "figures" / "fig1_consensus_vs_density.png").exists()
    assert (tmp_path / "figures" / "fig2_cost_vs_recovery.png").exists()
    assert (tmp_path / "figures" / "fig3_intelligibility_progression.png").exists()
