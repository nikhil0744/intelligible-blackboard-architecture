"""Offline integration fixtures; these do not establish model/Colab performance."""
import hashlib
import json
import re
import time
from dataclasses import replace
from pathlib import Path

import pytest

from agents.board_client import InMemoryBoard
from benchmarks.actions import BenchmarkAgent, active_root
from benchmarks.batch import configuration, resume_batch, run_batch
from benchmarks.data_tools import DataTools
from benchmarks.persistence import PersistenceError, RunStore
from benchmarks.preparation import PreparedBenchmark, prepare
from benchmarks.runtime import BenchmarkScheduler, Limits
from contracts.schemas import PXPTag
from llm_broker import LLMResponse, ModelBroker
from llm_broker.config import InferenceSettings


# Own tiny metric fixture: the actual pinned upstream evaluator is checked separately
# against downloaded assets. Ordinary CI neither downloads nor imports that project.
METRIC = b'''def str_to_float(value):
    return float(value)
def try_convert_to_number(value):
    return value
class Success(Metric):
    def __call__(self, predicted, target):
        return (float(predicted == target), 0)
'''


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    import benchmarks.preparation as module
    workload = [{"id": f"task-{i}", "query": f"Add the value column for sample {i}.", "answer": 6,
                 "answer_type": "numeric_exact", "data_sources": ["input.csv"],
                 "subtasks": [{"answer": "HIDDEN-SUBTASK"}], "solution": "HIDDEN-SOLUTION"} for i in range(3)]
    blobs = {"workload/legal.json": json.dumps(workload).encode(), "benchmark/metrics.py": METRIC,
             "data/input.csv": b"name,value\na,2\nb,4\n"}
    spec = {"repo": "fixture/test", "revision": "pinned-fixture", "profile": "fixture-numeric-v1",
            "workload": "workload/legal.json", "task_ids": [r["id"] for r in workload],
            "coverage_note": "explicit offline fixture", "assets": [
                {"path": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)} for name, data in blobs.items()]}
    monkeypatch.setattr(module, "resources", lambda: {"kramabench": spec})
    fetches = []
    def fetch(url):
        name = url.split("/pinned-fixture/")[1]
        fetches.append(name)
        return blobs[name]
    cache = tmp_path / "cache"
    root = prepare(cache, fetch=fetch)
    prepare(cache, fetch=lambda url: pytest.fail("valid cache must be reused"))
    assert len(fetches) == len(blobs)
    return PreparedBenchmark.open(cache)


def decision(kind, tag, claim, target=None):
    value = {"kind": kind, "tag": tag, "target_contribution_id": target,
             "prediction": {"claim": str(claim), "confidence": .8},
             "explanation": {"rationale": "Computed using the permitted CSV.", "evidence": ["value column"]}}
    if kind == "final":
        value["output"] = claim
    return value


class ScriptedBackend:
    name = "scripted-test"
    def __init__(self, answer=6):
        self.answer, self.calls, self.prompts = answer, [], []

    def complete(self, request, model):
        self.prompts.append([m.content for m in request.messages])
        self.calls.append(request.agent_id)
        user = request.messages[1].content
        root = re.search(r"Current claim root id=([^ .]+)", request.messages[0].content)
        if root:
            action = decision("final", "RATIFY", self.answer, root[1])
        elif request.json_schema.get("title") == "ToolAction":
            action = {"kind": "tool", "name": "query", "arguments": {
                "file": "input.csv", "aggregate": {"column": "value", "op": "sum"}}}
        else:
            assert '"value": 6.0' in user
            action = decision("final", "REVISE", self.answer)
        return LLMResponse(text=json.dumps(action), model=model, backend=self.name,
                           prompt_tokens=20, completion_tokens=10)


SETTINGS = InferenceSettings(model="fixture-model", max_repairs=0, max_retries=0)
READINESS = {"model_digest": "fixture-digest", "backend_version": "fixture", "ok": True}


def broker(backend=None):
    return ModelBroker(backend or ScriptedBackend(), SETTINGS.model, max_concurrency=1, max_retries=0)


def test_actual_agents_tools_grade_and_per_attempt_usage(prepared, tmp_path):
    model = broker()
    store, summary = run_batch(prepared, model, SETTINGS, tmp_path / "results", readiness=READINESS, observer=lambda e: None)
    assert summary["selected"] == summary["scored"] == summary["correct"] == 3
    assert summary["consensus"] == 3
    for result in summary["outcomes"]:
        assert result["turns"] == 4  # REVISE plus three distinct RATIFY votes
        assert result["tool_rounds"] == 1
        assert result["usage"]["calls"] == 5
        assert result["usage"]["total_tokens"] == 150
        assert result["recovery"]["calls"] == 0
        path = store.root / "attempts" / result["attempt_id"]
        events = [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]
        assert next(i for i, e in enumerate(events) if e["kind"] == "output_frozen") < next(i for i, e in enumerate(events) if e["kind"] == "evaluation")
        text = (path / "events.jsonl").read_text()
        assert "HIDDEN-SUBTASK" not in text and "HIDDEN-SOLUTION" not in text
    assert summary["usage_all_attempts"]["total_tokens"] == 450
    assert RunStore(store.root).summarize() == summary


def test_duplicate_inspection_then_query_yields_to_panel(prepared, tmp_path):
    class LoopingBackend(ScriptedBackend):
        def complete(self, request, model):
            if request.json_schema.get("title") == "ToolAction" and "duplicate" not in request.messages[0].content:
                return LLMResponse(text=json.dumps({"kind": "tool", "name": "inspect",
                                    "arguments": {"file": "input.csv"}}), model=model, backend=self.name)
            if request.json_schema.get("title") == "OpeningSubmission":
                assert '"value": 6.0' in request.messages[1].content
            return super().complete(request, model)

    model = broker(LoopingBackend())
    events = []
    store, _ = run_batch(prepared, model, SETTINGS, tmp_path / "results", limit=1,
                         readiness=READINESS, observer=events.append)
    result = store.attempts()[0]
    assert result["tool_rounds"] == 2
    assert result["turns"] == 4
    assert result["consensus"] and result["grade"]["correct"]
    assert {"analyst", "critic", "reasoner"} <= set(model.backend.calls)
    order = [event["kind"] for event in events]
    assert order.index("tool_request_blocked") < order.index("evidence_ready") < order.index("contribution")


@pytest.mark.parametrize("inspect_header", [0, True, False, None])
def test_query_inherits_suggested_header_and_accepts_year_membership(prepared, tmp_path, inspect_header):
    # Title rows and model requests mirror the live failure on the official CSV.
    (prepared.root / "data/input.csv").write_text(
        "Title,,\n,,\nReports,,\nYear,Contributor,Reports\n"
        "2022,Web,2\n2023,Web,4\n2024,Mobile,100\n2021,Web,100\n"
    )
    query = {"kind": "tool", "name": "query", "arguments": {
        "file": "input.csv",
        "filters": [{"column": "Year", "op": "in", "value": ["2022", "2023", "2024"]},
                    {"column": "Contributor", "op": "eq", "value": "Web"}],
        "aggregate": {"column": "Reports", "op": "sum"},
    }}
    opening = iter([
        {"kind": "tool", "name": "inspect", "arguments": {"file": "input.csv", "header": inspect_header}},
        query,
    ])

    class HeaderBackend(ScriptedBackend):
        def complete(self, request, model):
            if request.json_schema.get("title") == "ToolAction":
                return LLMResponse(text=json.dumps(next(opening)), model=model, backend=self.name)
            return super().complete(request, model)

    events = []
    store, _ = run_batch(prepared, broker(HeaderBackend()), SETTINGS, tmp_path / "results",
                         limit=1, readiness=READINESS, observer=events.append)
    requests = [e for e in events if e["kind"] == "tool_requested" and e["name"] == "query"]
    assert [e["arguments"]["header"] for e in requests] == [3]
    assert requests[-1]["header_inherited"] is True
    assert store.attempts()[0]["grade"]["correct"]


@pytest.mark.parametrize("action,status", [
    (decision("contribution", "REVISE", 123456), "invalid_output"),
    ({"kind": "tool", "name": "list_files", "arguments": {}}, "insufficient_evidence"),
    ({"kind": "tool", "name": "calculate", "arguments": {"values": [float("inf")]}}, "invalid_output"),
])
def test_opening_cannot_post_without_csv_query(prepared, tmp_path, action, status):
    class NoEvidenceBackend(ScriptedBackend):
        def complete(self, request, model):
            assert request.json_schema.get("title") == "ToolAction"
            assert "use REVISE" not in request.messages[1].content
            return LLMResponse(text=json.dumps(action), model=model, backend=self.name)

    store, _ = run_batch(prepared, broker(NoEvidenceBackend()), SETTINGS, tmp_path / "results",
                         limit=1, readiness=READINESS, observer=lambda e: None)
    result = store.attempts()[0]
    assert result["status"] == status
    assert result["turns"] == 0 and result["output"] is None
    assert result["grade"]["status"] == "no_final_submission"


def test_repeated_disagreement_stops_before_turn_budget(prepared, tmp_path):
    class StalledBackend(ScriptedBackend):
        def complete(self, request, model):
            root = re.search(r"Current claim root id=([^ .]+)", request.messages[0].content)
            if root is None:
                return super().complete(request, model)
            tag = "REFUTE" if request.agent_id == "critic" else "REVISE"
            action = decision("contribution", tag, self.answer, root[1])
            return LLMResponse(text=json.dumps(action), model=model, backend=self.name)

    store, summary = run_batch(prepared, broker(StalledBackend()), SETTINGS, tmp_path / "results",
                               limit=1, readiness=READINESS, observer=lambda e: None)
    result = store.attempts()[0]
    assert result["status"] == "stalled" and result["turns"] < Limits().max_turns
    assert not result["consensus"] and summary["stalled"] == 1


def test_interrupt_after_commit_resume_preserves_wrong_answers(prepared, tmp_path):
    model = broker(ScriptedBackend(answer=5))
    saved = []
    def stop(event):
        if event["kind"] == "attempt_saved":
            saved.append(event)
            raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        run_batch(prepared, model, SETTINGS, tmp_path / "results", readiness=READINESS, observer=stop)
    root = Path(saved[0]["run_dir"])
    before = RunStore(root).attempts()[0]
    store, summary = resume_batch(root, prepared, model, SETTINGS, readiness=READINESS, observer=lambda e: None)
    assert summary["attempts"] == 3
    assert summary["incorrect"] == 3 and summary["correct"] == 0
    assert store.attempts()[0]["attempt_id"] == before["attempt_id"]
    assert len(model.backend.calls) == 15
    # A second resume launches no calls, and incorrect results are still skipped.
    resume_batch(root, prepared, model, SETTINGS, readiness=READINESS, observer=lambda e: None)
    assert len(model.backend.calls) == 15
    with pytest.raises(ValueError, match="Cannot resume"):
        resume_batch(root, prepared, model, replace(SETTINGS, temperature=.9), readiness=READINESS)


def test_partial_attempt_retained_and_commit_corruption_rejected(prepared, tmp_path):
    model = broker(ScriptedBackend())
    limits = Limits()
    config = configuration(prepared, SETTINGS, limits, 0, READINESS)
    store = RunStore.create(tmp_path / "run", config, [t.model_dump(mode="json") for t in prepared.tasks()],
                            {"storage": "persistent"})
    partial = store.start("task-0")
    partial.event("fixture_interrupted")
    (partial.path / "calls.jsonl").unlink()  # interrupted during recorder initialization
    _, summary = resume_batch(store.root, prepared, model, SETTINGS, readiness=READINESS, observer=lambda e: None)
    assert summary["attempts"] == 4
    assert summary["partial_call_records"] == 1
    assert summary["usage_all_attempts"]["total_tokens"] is None
    assert partial.path.exists() and not (partial.path / "complete.json").exists()
    committed = next(p for p in (store.root / "attempts").iterdir() if (p / "complete.json").exists())
    (committed / "events.jsonl").write_text("tampered\n")
    with pytest.raises(PersistenceError, match="checksum"):
        store.summarize()


def test_isolated_tools_and_timeout_kills_worker(tmp_path, monkeypatch):
    import benchmarks.data_tools as module
    root = tmp_path / "inputs"
    root.mkdir()
    (root / "input.csv").write_text("preamble,,\nlabel,value,x\na,2,0\nb,4,0\n")
    secret = tmp_path / "answers.csv"
    secret.write_text("answer\n6\n")
    (root / "link.csv").symlink_to(secret)
    tools = DataTools(root, ("input.csv", "link.csv"))
    assert not tools.execute("bad", "inspect", {"file": "../answers.csv"})["ok"]
    assert not tools.execute("link", "inspect", {"file": "link.csv"})["ok"]
    result = tools.execute("sum", "query", {"file": "input.csv", "header": 1, "aggregate": {"op": "sum", "column": "value"}})
    assert result["data"]["value"] == 6
    assert tools.execute("sum", "query", {"file": "input.csv", "header": 1, "aggregate": {"op": "sum", "column": "value"}}) == result
    with pytest.raises(ValueError, match="reused"):
        tools.execute("sum", "list_files", {})
    def hang(*args):
        time.sleep(10)
        (tmp_path / "escaped").touch()
    monkeypatch.setattr(module, "_operation", hang)
    tools.timeout_s = .05
    assert tools.execute("timeout", "list_files", {})["timed_out"]
    time.sleep(.1)
    assert not (tmp_path / "escaped").exists()


def test_incompatible_votes_revision_and_unknown_usage(prepared, tmp_path):
    agent = BenchmarkAgent("analyst", "data_analyst", broker(), **SETTINGS.agent_kwargs())
    board = InMemoryBoard()
    snap = board.create_session("t", "question", session_id="s")
    first = agent.to_contribution(snap, agent.parse_output(json.dumps(decision("final", "REVISE", 6))), final_submission=True, final_output=6)
    snap = board.submit(first)
    with pytest.raises(Exception, match="not the claim"):
        agent.validate_output(agent.parse_output(json.dumps(decision("final", "RATIFY", 5, first.contribution_id))), snap)
    sched = BenchmarkScheduler("s", board)
    for name in ("analyst", "critic", "reasoner"):
        sched.register_agent(name)
    for name in ("critic", "reasoner"):
        d = agent.parse_output(json.dumps(decision("final", "RATIFY", 6, first.contribution_id)))
        c = agent.to_contribution(snap, d)
        c.agent_id = name
        snap = board.submit(c)
    assert not sched._check_consensus(snap)  # author has not ratified
    revision = agent.to_contribution(snap, agent.parse_output(json.dumps(decision("final", "REVISE", 7))), final_submission=True, final_output=7)
    snap = board.submit(revision)
    assert not sched._check_consensus(snap)
    with pytest.raises(Exception, match="current REVISE"):
        agent.validate_output(agent.parse_output(json.dumps(decision("final", "RATIFY", 6, first.contribution_id))), snap)
    with pytest.raises(Exception, match="boolean"):
        agent.parse_output(json.dumps(decision("final", "REVISE", False)))


def test_failed_cache_and_missing_model_have_no_fallback(prepared, tmp_path):
    from benchmarks.batch import check_readiness
    (prepared.root / "data/input.csv").write_text("broken")
    from benchmarks.preparation import verify
    with pytest.raises(ValueError, match="corrupt"):
        verify(prepared.root, prepared.spec)
    with pytest.raises(ValueError, match="no mock"):
        check_readiness(broker(), SETTINGS, cpu_smoke=True)


def test_reference_changes_do_not_reach_prompts_or_inputs(prepared, tmp_path, monkeypatch):
    from benchmarks.runtime import SessionRunner
    from benchmarks.persistence import Attempt
    monkeypatch.setattr("benchmarks.runtime.uuid4", lambda: type("ID", (), {"hex": "fixture-request"})())
    captures = []
    for index, label in enumerate(["HIDDEN-LABEL-ONE", "HIDDEN-LABEL-TWO"]):
        workload = prepared.root / "workload/legal.json"
        records = json.loads(workload.read_text())
        for record in records:
            record["answer"] = label
            record["subtasks"] = [{"answer": label}]
            record["eval_MRN"] = label
        workload.write_text(json.dumps(records))
        task = prepared.tasks()[0]
        path = tmp_path / f"inputs-{index}"
        prepared.copy_inputs(task, path)
        assert sorted(p.name for p in path.iterdir()) == ["input.csv"]
        attempt = Attempt(tmp_path / f"attempt-{index}", {"session_id": "fixed-session", "attempt_id": str(index)})
        model = broker(ScriptedBackend())
        result = SessionRunner(model, SETTINGS, Limits(max_turns=1)).execute(task, path, attempt)
        assert result["status"] == "turn_limit" and result["output"] == 6
        captures.append(model.backend.prompts)
    assert captures[0] == captures[1]
    assert "HIDDEN-LABEL" not in json.dumps(captures)


def test_model_failure_unknown_usage_and_explicit_retry(prepared, tmp_path):
    from llm_broker import LLMError
    class FailedBackend:
        name = "scripted-test"
        def complete(self, request, model):
            raise LLMError("fixture server unavailable")
    failed = broker(FailedBackend())
    store, summary = run_batch(prepared, failed, SETTINGS, tmp_path / "results", limit=1,
                               readiness=READINESS, observer=lambda e: None)
    assert summary["errored"] == 1 and summary["scored"] == 0
    assert summary["usage_all_attempts"]["total_tokens"] is None
    assert summary["usage_all_attempts"]["unknown_usage_calls"] == 1
    successful = broker(ScriptedBackend())
    _, skipped = resume_batch(store.root, prepared, successful, SETTINGS, readiness=READINESS, observer=lambda e: None)
    assert skipped["attempts"] == 1 and successful.backend.calls == []
    _, retried = resume_batch(store.root, prepared, successful, SETTINGS, readiness=READINESS,
                              retry_failures=True, observer=lambda e: None)
    assert retried["attempts"] == 2 and retried["correct"] == 1
    assert retried["usage_all_attempts"]["total_tokens"] is None


def test_saved_trace_uses_actual_events_and_escapes_script_output(prepared, tmp_path):
    from benchmarks.presentation import export_viewer, read_trace, trace_table
    class MarkupBackend(ScriptedBackend):
        def complete(self, request, model):
            response = super().complete(request, model)
            action = json.loads(response.text)
            if action["kind"] == "tool":
                return response
            action["explanation"]["rationale"] = '</script><script>window.untrusted=true</script>'
            return response.model_copy(update={"text": json.dumps(action)})
    store, _ = run_batch(prepared, broker(MarkupBackend()), SETTINGS, tmp_path / "results",
                         limit=1, readiness=READINESS, observer=lambda e: None)
    _, result, events = read_trace(store.root)
    rows = trace_table(store.root)
    assert len([r for r in rows if r["kind"] == "contribution"]) == result["turns"]
    viewer = export_viewer(store.root, Path(__file__).parents[1] / "visualizer.html")
    trace = json.loads(viewer.with_name("trace.json").read_text())
    assert trace["mode"] == "recorded_benchmark" and trace["sandbox"] is None
    assert [s["contribution_id"] for s in trace["steps"] if "contribution_id" in s] == [
        e["contribution"]["contribution_id"] for e in events if e["kind"] == "contribution"]
    assert not any(s["sandboxVisible"] for s in trace["steps"])
    assert trace["result"]["recovery"]["calls"] == 0
    injection = viewer.read_text().split("<script>", 1)[1].split("</script>", 1)[0]
    assert "<script>window.untrusted" not in injection and "\\u003c/script\\u003e" in injection


def test_persistence_failure_stops_before_next_task(prepared, tmp_path, monkeypatch):
    import benchmarks.persistence as module
    original = module.write_json
    def fail_commit(path, value):
        if path.name == "complete.json":
            raise PersistenceError("fixture Drive full")
        return original(path, value)
    monkeypatch.setattr(module, "write_json", fail_commit)
    model = broker(ScriptedBackend())
    with pytest.raises(PersistenceError, match="Drive full"):
        run_batch(prepared, model, SETTINGS, tmp_path / "results", readiness=READINESS, observer=lambda e: None)
    assert len(model.backend.calls) == 5


def test_year_only_and_lookalike_queries_need_refinement(tmp_path):
    (tmp_path / "input.csv").write_text(
        "Title,,\n,,\nYear,Contributor,Reports\n"
        "2022,Web Fraud,2\n2022,Web IDT,99\n2023,Web Fraud,4\n2023,Mobile Fraud,30\n"
    )
    question = "How many frauds were reported over the web between 2022 and 2023?"
    tools = DataTools(tmp_path, ("input.csv",), question=question)
    years = {"column": "Year", "op": "in", "value": [2022, 2023]}
    for index, (label, issues) in enumerate([(None, True), ("Web IDT", True), ("Web Fraud", False)]):
        filters = [years]
        if label:
            filters.append({"column": "Contributor", "op": "eq", "value": label})
        result = tools.execute(str(index), "query", {"file": "input.csv", "filters": filters,
                               "aggregate": {"op": "sum", "column": "Reports"}})
        assert result["ok"] and bool(result["data"].get("scope_issues")) == issues
    assert result["data"]["value"] == 6


def test_all_categories_ignores_similarly_named_footer(tmp_path):
    (tmp_path / "input.csv").write_text(
        "Payment Method, Total $ Loss \nCard,$2M\nCash,$3M\n# of Reports with Payment Method,25% of Total\n"
    )
    tools = DataTools(tmp_path, ("input.csv",), question="What is the total loss over all payment methods?")
    result = tools.execute("all", "query", {"file": "input.csv",
                           "aggregate": {"column": "total $ loss", "op": "sum"}})
    assert result["ok"] and result["data"]["value"] == 5
    assert not result["data"].get("scope_issues")
