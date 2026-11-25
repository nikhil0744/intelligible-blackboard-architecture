import json

import pytest

from benchmarks.preparation import PreparedBenchmark, digest, resources


def test_expanded_manifest_separates_pilot_and_has_20_tasks():
    manifest = resources()
    expanded = manifest["kramabench20"]
    pilot = manifest["kramabench"]
    assert len(set(expanded["task_ids"])) == 20
    assert expanded["task_ids"][:3] == pilot["task_ids"] == expanded["strict_scope_task_ids"]
    assert expanded["revision"] == pilot["revision"]
    assert expanded["profile"] != pilot["profile"]
    assert len({asset["path"] for asset in expanded["assets"] if asset["path"].endswith(".csv")}) == 15


def test_multifile_assets_and_typed_metric_dispatch(tmp_path):
    csv_paths = ["legal/input/subdir/one.csv", "legal/input/two.csv"]
    for name in csv_paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("State,Count\nSomewhere,2\n")
    workload = tmp_path / "workload.json"
    workload.write_text(json.dumps([
        {"id": "list", "answer_type": "list_exact", "query": "Where?", "answer": ["Somewhere"],
         "data_sources": ["subdir/one.csv", "two.csv"]},
        {"id": "ratio", "answer_type": "numeric_approximate", "query": "What ratio?", "answer": 2.0,
         "data_sources": ["two.csv"]},
    ]))
    metrics = tmp_path / "benchmark/metrics.py"
    metrics.parent.mkdir()
    metrics.write_text(
        "def str_to_float(value): return float(value)\n"
        "def try_convert_to_number(value): return value\n"
        "class F1(Metric):\n"
        "    def __call__(self, predicted, target): return (1.0 if predicted == target else 0.5,)\n"
        "class Precision(F1): pass\n"
        "class Recall(F1): pass\n"
        "class RAEScore(Metric):\n"
        "    def __call__(self, predicted, target): return (1.0 if predicted == target else 0.5,)\n"
        "class MeanAbsoluteError(RAEScore): pass\n"
        "class MeanSquaredError(RAEScore): pass\n"
    )
    spec = {"assets": [{"path": name, "sha256": digest(tmp_path / name)}
                       for name in [*csv_paths, "benchmark/metrics.py"]],
            "strict_scope_task_ids": [], "task_ids": ["list", "ratio"],
            "answer_types": ["list_exact", "numeric_approximate"], "workload": workload.name,
            "revision": "test-audited-metrics"}
    prepared = PreparedBenchmark(tmp_path, spec)
    listing, ratio = prepared.tasks()
    assert listing.files == ("one.csv", "two.csv")
    assert listing.scope_mode == ratio.scope_mode == "advisory"
    assert "Somewhere" not in listing.model_dump_json()
    target = tmp_path / "isolated-input"
    prepared.copy_inputs(listing, target)
    assert {p.name for p in target.iterdir()} == {"one.csv", "two.csv"}

    partial = prepared.evaluate("list", ["Elsewhere"])
    assert partial["status"] == "scored" and partial["metric"] == "f1"
    assert partial["score"] == .5 and not partial["correct"]
    assert set(partial["metrics"]) == {"f1", "precision", "recall"}
    with pytest.raises(ValueError, match="requires lists"):
        prepared.evaluate("list", '["Somewhere"]')
    full = prepared.evaluate("ratio", 2.0)
    assert full["correct"] and full["metric"] == "rae_score"
    assert set(full["metrics"]) == {"rae_score", "mean_absolute_error", "mean_squared_error"}


def test_table_operations_can_rank_categories_and_filter_calculated_shares(tmp_path):
    from benchmarks.data_tools import DataTools

    (tmp_path / "rows.csv").write_text(
        "State,Category,Reports,Other\n"
        "A,Fraud,2,4\nA,Identity,3,2\nB,Fraud,6,1\nB,Identity,5,2\n"
    )
    tools = DataTools(tmp_path, ("rows.csv",))
    ranked = tools.execute("rank", "query", {
        "file": "rows.csv", "first_per": "State",
        "sort": {"column": "Reports", "descending": True, "numeric": True},
        "filters": [{"column": "Category", "op": "eq", "value": "Identity"}],
        "select": ["State"],
    })
    assert ranked["data"]["rows"] == [{"State": "A"}]
    shares = tools.execute("shares", "query", {
        "file": "rows.csv", "derive": {"name": "share", "op": "share", "columns": ["Reports", "Other"]},
        "filters": [{"column": "share", "op": "ge", "value": 0.5}],
        "offset": 1, "limit": 1, "select": ["Reports"],
    })
    assert shares["data"]["rows"] == [{"Reports": "6"}]
    assert shares["data"]["truncated"]


def test_later_csv_section_can_be_inspected_and_queried(tmp_path):
    from benchmarks.data_tools import DataTools

    (tmp_path / "sections.csv").write_text("Title,,\n" + ",,\n" * 21 + "Range,Reports,\nLow,3,\nHigh,5,\n")
    tools = DataTools(tmp_path, ("sections.csv",))
    inspected = tools.execute("inspect", "inspect", {"file": "sections.csv", "header": 0})
    assert 22 in inspected["data"]["header_candidates"]
    result = tools.execute("later", "query", {
        "file": "sections.csv", "header": 22, "aggregate": {"op": "sum", "column": "Reports"},
    })
    assert result["data"]["value"] == 8
