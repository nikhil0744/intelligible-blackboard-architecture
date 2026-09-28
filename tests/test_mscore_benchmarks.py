"""Tests for MSCoRe dataset ingestion and the shared batch trial runner."""

import json
from pathlib import Path
import pytest

from benchmarks.mscore import MSCoReIngestor, MSCoReItem, BatchTrialRunner
from contracts.schemas import BenchmarkType, BlackboardStatus


def test_mscore_sample_dataset_integrity():
    items = MSCoReIngestor.get_sample_dataset()
    assert len(items) >= 4
    for item in items:
        assert item.id
        assert item.question
        assert item.ground_truth
        assert len(item.choices) >= 2


def test_mscore_load_from_json(tmp_path: Path):
    raw_data = [
        {
            "id": "q1",
            "question": "What is 2+2?",
            "choices": ["3", "4", "5"],
            "ground_truth": "4",
            "domain": "math",
        },
        {
            "task_id": "q2",
            "prompt": "Capital of France?",
            "options": ["Berlin", "Paris", "Rome"],
            "answer": "Paris",
            "category": "geography",
        },
    ]
    json_file = tmp_path / "test_mscore.json"
    json_file.write_text(json.dumps(raw_data), encoding="utf-8")

    items = MSCoReIngestor.load_from_file(json_file)
    assert len(items) == 2
    assert items[0].id == "q1"
    assert items[0].ground_truth == "4"
    assert items[1].id == "q2"
    assert items[1].ground_truth == "Paris"


def test_mscore_to_snapshot():
    item = MSCoReItem(
        id="item_42",
        question="Should thrombolysis be administered?",
        ground_truth="Yes",
    )
    snapshot = MSCoReIngestor.to_blackboard_snapshot(item, session_id="test_sess_42")
    assert snapshot.session_id == "test_sess_42"
    assert snapshot.task_id == "item_42"
    assert snapshot.task_description == "Should thrombolysis be administered?"
    assert snapshot.status == BlackboardStatus.ACTIVE


@pytest.mark.asyncio
async def test_batch_trial_runner_matrix_and_execution(tmp_path: Path):
    items = MSCoReIngestor.get_sample_dataset()[:2]
    runner = BatchTrialRunner(streaming_enabled=False)

    # Generate 2 items * 4 densities = 8 configs
    configs = runner.generate_ablation_matrix(items, densities=[0.0, 0.33, 0.66, 1.0])
    assert len(configs) == 8

    # Run batch
    results = await runner.run_batch(configs)
    assert len(results) == 8
    for r in results:
        assert r.turns_taken > 0
        assert r.total_token_cost > 0
        assert r.duration_seconds >= 0.0
        assert r.intelligibility.level is not None

    # Test export to JSON
    json_path = tmp_path / "results.json"
    runner.export_results_to_json(results, json_path)
    assert json_path.exists()
    assert len(json.loads(json_path.read_text(encoding="utf-8"))) == 8

    # Test export to CSV
    csv_path = tmp_path / "results.csv"
    runner.export_results_to_csv(results, csv_path)
    assert csv_path.exists()
