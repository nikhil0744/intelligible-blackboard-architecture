"""Regression checks for the bounded presentation path, independent of legacy recovery."""
import asyncio
import json
from pathlib import Path
import re
import threading
import urllib.error
import urllib.request

import pytest

from contracts.schemas import AgentContribution, PEXPayload, Prediction, Explanation, PXPTag
from llm_broker import InferenceSettings, ModelBroker, LLMRequest, ChatMessage
from llm_broker.backends.mock import MockBackend
from llm_broker.budget import BudgetExceeded, CallBudget, budget_scope
from scripts import prototype_artifacts
from scripts.prototype_run import (
    AgreementTracker, PrototypeTask, TrialOutcome, grade, load_inputs, run_prototype,
)
from scripts.prototype_demo import make_server


@pytest.fixture(autouse=True)
def no_repeated_plotting(monkeypatch):
    monkeypatch.setattr(prototype_artifacts, "plot_results", lambda *a: None)


def answer(req):
    text = "\n".join(m.content for m in req.messages if m.role == "user")
    ids = re.findall(r"\[id=([^\]]+)\]", text)
    return json.dumps({"tag": "RATIFY" if ids else "REVISE",
        "target_contribution_id": ids[0] if ids else None,
        "prediction": {"claim": "5", "confidence": 0.5},
        "explanation": {"rationale": "Each machine makes a widget in five minutes.", "evidence": ["CALC: 5 * 100 / 100 = 5"]}})


def run(tmp_path, *, tasks=None, references=None, responder=answer, backend=None, **kwargs):
    broker = ModelBroker(backend or MockBackend(responder=responder), "test-model", max_retries=2, backoff_s=0)
    settings = InferenceSettings(backend="mock", model="test-model")
    result = run_prototype(tasks or [PrototypeTask(task_id="machines", question="Real unique widget question?")],
        references or {}, tmp_path / "run", settings=settings, seeds=[42], broker=broker, **kwargs)
    return result, broker


def contribution(cid, agent, tag, claim="5", target=None):
    return AgentContribution(contribution_id=cid, session_id="s", turn_index=0, agent_id=agent,
        tag=tag, target_contribution_id=target,
        payload=PEXPayload(prediction=Prediction(claim=claim, confidence=.5),
                           explanation=Explanation(rationale="Reason", evidence=[])))


def test_real_task_agreement_correctness_artifacts_and_accounting(tmp_path):
    result, broker = run(tmp_path, references={"machines": 5})
    r = result["results"][0]
    assert result["exit_code"] == 0
    assert r["answer_agreement"] and r["correct"]
    assert r["turns_succeeded"] == 4  # the original author must explicitly ratify too
    rows = prototype_artifacts.read_jsonl(tmp_path / "run" / "trace.jsonl")
    posts = [e["contribution"] for e in rows if e["type"] == "contribution"]
    assert [c["turn_index"] for c in posts] == [0, 1, 2, 3]
    calls = broker.ledger.records(trial_id=r["trial_id"])
    assert sum(c.prompt_tokens + c.completion_tokens for c in calls) == r["usage"]["total_tokens"]
    assert all("Real unique widget question?" in c.messages[-1].content for c in broker.backend.calls if c.agent_id != "_preflight")
    for path in ("manifest.json", "summary.csv", "summary.json", "demo.html", "calls.jsonl", "results.jsonl", "trace.jsonl"):
        assert (tmp_path / "run" / path).exists()


def test_wrong_unanimous_answer_stays_wrong(tmp_path):
    result, _ = run(tmp_path, references={"machines": "4"})
    assert result["results"][0]["answer_agreement"]
    assert result["results"][0]["correct"] is False


def test_corrected_proposal_can_receive_fresh_explicit_votes(tmp_path):
    """A corrected answer must be endorsed afresh, using the active proposal id."""
    ordinary = 0
    def correct_then_endorse(req):
        nonlocal ordinary
        if req.agent_id == "_preflight":
            return '{"ok":true}'
        ordinary += 1
        active = re.search(r"The active proposal is id=([^,]+), claim=", req.messages[0].content)
        return json.dumps({"tag": "REVISE" if ordinary <= 2 else "RATIFY",
            "target_contribution_id": active[1] if active else None,
            "prediction": {"claim": "0.10" if ordinary == 1 else "0.05", "confidence": .8},
            "explanation": {"rationale": "Corrected by checking the sum and difference.", "evidence": [
                "CALC: 1.10 - 1.00 = 0.10" if ordinary == 1 else "CALC: (1.10 - 1.00) / 2 = 0.05"]}})
    result, broker = run(tmp_path, tasks=[PrototypeTask(task_id="bat", question="Bat and ball question")],
        references={"bat": "0.05"}, responder=correct_then_endorse)
    r = result["results"][0]
    assert r["correct"] and r["answer_agreement"] and r["turns_succeeded"] == 5
    assert r["usage"]["repairs"] == 0
    rows = prototype_artifacts.read_jsonl(tmp_path / "run" / "trace.jsonl")
    posts = [row for row in rows if row["type"] == "contribution"]
    assert not posts[1]["agreeing_agents"]
    assert all(row["contribution"]["target_contribution_id"] == posts[1]["contribution"]["contribution_id"]
               for row in posts[2:])
    assert all("0.05" not in m.content for c in broker.backend.calls
               if c.agent_id == "general_reasoner" and not re.search(r"\[id=", c.messages[-1].content)
               for m in c.messages)


def test_repeated_identical_revisions_are_not_silently_counted_as_agreement(tmp_path):
    def repeat_revision(req):
        return json.dumps({"tag": "REVISE", "target_contribution_id": None,
            "prediction": {"claim": "0.05", "confidence": .8},
            "explanation": {"rationale": "Repeated proposal", "evidence": ["CALC: (1.10 - 1.00) / 2 = 0.05"]}})
    result, _ = run(tmp_path, references={"machines": "0.05"}, responder=repeat_revision, max_turns=6)
    r = result["results"][0]
    assert r["outcome"] == "turn_limit" and r["turns_succeeded"] == 6
    assert r["correct"] and not r["answer_agreement"] and r["provisional"]


def test_prototype_requests_reasoning_before_prediction_and_tag(tmp_path):
    from prompts.schema import decision_json_schema, AgentDecision
    result, broker = run(tmp_path, references={"machines": 5})
    requests = [c for c in broker.backend.calls if c.agent_id != "_preflight"]
    order = ["explanation", "prediction", "tag", "target_contribution_id"]
    assert all(list(c.json_schema["properties"]) == order for c in requests)
    assert result["manifest"]["prompt_policy"] == "prototype-v6-calculation-chains"
    assert all(a["reasoning_first"] for a in result["manifest"]["agents"])
    original = decision_json_schema()
    reordered = decision_json_schema(reasoning_first=True)
    assert original["properties"] == reordered["properties"]
    assert set(original["required"]) == set(reordered["required"])
    parsed = AgentDecision.model_validate_json(answer(requests[0]))
    assert parsed.prediction.claim == "5" and parsed.tag == PXPTag.REVISE


def test_explanation_alone_is_not_treated_as_a_corrected_prediction(tmp_path):
    """Preserve inconsistent model output as failure; do not extract a preferred answer."""
    def stuck(req):
        if req.agent_id == "_preflight": return '{"ok":true}'
        active = re.search(r"The active proposal is id=([^,]+), claim=", req.messages[0].content)
        return json.dumps({"tag": "REFUTE" if active else "REVISE",
            "target_contribution_id": active[1] if active else None,
            "prediction": {"claim": "0.10", "confidence": .5},
            "explanation": {"rationale": "My calculation gives 0.05, rather than 0.10.", "evidence": ["CALC: 1.10 - 1.00 = 0.10"]}})
    result, _ = run(tmp_path, references={"machines": "0.05"}, responder=stuck)
    r = result["results"][0]
    assert r["final_answer"] == "0.10" and r["correct"] is False
    assert not r["answer_agreement"] and r["outcome"] == "disagreement_stall"


def test_stale_endorsement_cannot_redirect_the_next_agents_leading_claim(tmp_path):
    turns = 0
    def revise_and_revisit_old_post(req):
        nonlocal turns
        if req.agent_id == "_preflight": return '{"ok":true}'
        turns += 1
        active = re.search(r"The active proposal is id=([^,]+), claim=", req.messages[0].content)
        ids = re.findall(r"\[id=([^\]]+)\]", req.messages[-1].content)
        claim = "0.10" if turns in (1, 3) else "0.05"
        target = ids[0] if turns == 3 else (active[1] if active else None)
        if turns == 4:
            assert "stale" in req.messages[-1].content.lower() or "superseded" in req.messages[-1].content
            assert any("CURRENT LEADING CLAIM: 0.05" in m.content for m in req.messages)
        return json.dumps({"tag": "REVISE" if turns <= 2 else "RATIFY",
            "target_contribution_id": target,
            "prediction": {"claim": claim, "confidence": .8},
            "explanation": {"rationale": "Position for this test", "evidence": [f"CALC: {claim} = {claim}"]}})
    result, _ = run(tmp_path, references={"machines": "0.05"}, responder=revise_and_revisit_old_post)
    r = result["results"][0]
    assert r["correct"] and r["answer_agreement"] and r["turns_succeeded"] == 5
    assert r["usage"]["repairs"] == 1
    rows = prototype_artifacts.read_jsonl(tmp_path / "run" / "trace.jsonl")
    posts = [row for row in rows if row["type"] == "contribution"]
    assert posts[2]["contribution"]["payload"]["prediction"]["claim"] == "0.05"
    assert "stale_ratification" in posts[2]["contribution"]["metadata"]["repair_reasons"][0]


def test_reference_changes_do_not_change_prompts_or_predictions(tmp_path):
    a, ba = run(tmp_path / "a", references={"machines": "SECRET_ANSWER_A"})
    b, bb = run(tmp_path / "b", references={"machines": "SECRET_ANSWER_B"})
    def inputs(backend):
        return [re.sub(r"[0-9a-f]{8}-[0-9a-f-]{27}", "UUID", "\n".join(m.content for m in c.messages))
                for c in backend.calls]
    assert inputs(ba.backend) == inputs(bb.backend)
    assert a["results"][0]["final_answer"] == b["results"][0]["final_answer"]
    assert "SECRET_ANSWER" not in (tmp_path / "a" / "run" / "trace.jsonl").read_text()


def test_agreement_requires_three_distinct_agents_and_current_proposal():
    t = AgreementTracker(["A", "B", "C"])
    t.add(contribution("p", "A", PXPTag.REVISE))
    for agent in ("A", "B", "B"):
        t.add(contribution("r" + agent, agent, PXPTag.RATIFY, target="p"))
    assert not t.agreed
    t.add(contribution("q", "C", PXPTag.REVISE, claim="4"))
    t.add(contribution("old", "C", PXPTag.RATIFY, target="p"))
    assert not t.votes
    for agent in ("A", "B", "C"):
        t.add(contribution("new" + agent, agent, PXPTag.RATIFY, claim="4", target="q"))
    assert t.agreed
    t.add(contribution("crit", "B", PXPTag.REFUTE, target="q"))
    assert not t.agreed


def test_ratification_chain_and_conflicting_claim():
    t = AgreementTracker(["A", "B", "C"])
    t.add(contribution("p", "A", PXPTag.REVISE))
    t.add(contribution("a", "A", PXPTag.RATIFY, target="p"))
    t.add(contribution("b", "B", PXPTag.RATIFY, target="a"))
    t.add(contribution("bad", "C", PXPTag.RATIFY, claim="4", target="b"))
    assert not t.agreed
    t.add(contribution("c", "C", PXPTag.RATIFY, target="b"))
    assert t.agreed


@pytest.mark.parametrize("value", [0, False])
def test_falsey_answers_and_strict_grading(value):
    assert grade(str(value), "t", {"t": value}) == ("strict_match", True)
    assert grade("", "t", {"t": value}) == ("strict_match", False)
    assert grade(str(value) + " extra", "t", {"t": value}) == ("strict_match", False)
    assert grade("4", "t", {}) == ("unscored", None)
    assert grade("4", "t", {"t": {"manual_review": True}}) == ("manual_review", None)


@pytest.mark.parametrize("bad", [
    {"task_id": "x", "question": " "},
    {"task_id": "x", "question": "Q", "context": {"nested": [{"answer": "leak"}]}},
    {"task_id": "x", "question": "Q", "answer": "leak"},
])
def test_reject_bad_task_inputs(bad):
    with pytest.raises(ValueError):
        PrototypeTask.model_validate(bad)


def test_duplicate_tasks_and_invalid_refs(tmp_path):
    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps([{"task_id": "x", "question": "Q"}] * 2))
    with pytest.raises(ValueError, match="duplicate"):
        load_inputs(tasks, None)
    tasks.write_text('[{"task_id":"x","question":"Q"}]')
    refs = tmp_path / "refs.json"
    refs.write_text('{"x": []}')
    with pytest.raises(ValueError, match="invalid reference"):
        load_inputs(tasks, refs)


def test_per_trial_cost_does_not_accumulate(tmp_path):
    tasks = [PrototypeTask(task_id="a", question="Q"), PrototypeTask(task_id="b", question="Q")]
    result, _ = run(tmp_path, tasks=tasks)
    a, b = result["results"]
    assert a["usage"]["total_tokens"] == b["usage"]["total_tokens"]


def test_unknown_usage_stays_unknown(tmp_path):
    class Unknown(MockBackend):
        def complete(self, request, model):
            r = super().complete(request, model)
            r.prompt_tokens = r.completion_tokens = None
            return r
    result, _ = run(tmp_path, backend=Unknown(responder=answer))
    assert result["results"][0]["usage"]["total_tokens"] is None
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert summary["total_tokens"] is None and summary["unknown_usage_runs"] == 1


def test_malformed_responses_cannot_create_agreement(tmp_path):
    result, _ = run(tmp_path, responder=lambda req: "not JSON", max_turns=3)
    r = result["results"][0]
    assert r["outcome"] == "inference_failure"
    assert not r["answer_agreement"] and r["final_answer"] is None
    assert r["turns_failed"] == 3 and r["usage"]["repairs"] == 6


def test_call_budget_counts_retries_and_repairs(tmp_path):
    result, broker = run(tmp_path, responder=lambda req: "not JSON", max_calls=2)
    r = result["results"][0]
    assert r["outcome"] == "call_limit"
    assert r["usage"]["backend_attempts"] == 2
    assert len([r for r in broker.backend.calls if r.agent_id != "_preflight"]) == 2
    from llm_broker.types import LLMError
    class Fails(MockBackend):
        def complete(self, req, model):
            raise LLMError("backend offline")
    b = ModelBroker(Fails(), "m", max_retries=5, backoff_s=0)
    with budget_scope(CallBudget(max_attempts=2)):
        with pytest.raises(BudgetExceeded):
            b.generate(LLMRequest(messages=[ChatMessage(role="user", content="q")]))
    assert b.ledger.summary().backend_attempts == 2


def test_deadline_caps_transport_timeout():
    captured = []
    class Captures(MockBackend):
        def complete(self, req, model):
            captured.append(req.timeout_s)
            return super().complete(req, model)
    b = ModelBroker(Captures(), "m")
    with budget_scope(CallBudget(timeout_s=0.5)):
        b.generate(LLMRequest(messages=[ChatMessage(role="user", content="q")]))
    assert 0 < captured[0] <= 0.5


def test_late_model_result_does_not_become_success(tmp_path):
    result, _ = run(tmp_path, backend=MockBackend(responder=answer, delay_s=.02), timeout_s=.001)
    assert result["results"][0]["outcome"] == "time_limit"
    assert not result["results"][0]["answer_agreement"]


def test_partial_run_survives_cancellation_and_output_cannot_be_reused(tmp_path):
    count = 0
    def interrupt(req):
        nonlocal count
        if req.agent_id != "_preflight":
            count += 1
            if count == 6:
                raise KeyboardInterrupt
        return answer(req)
    tasks = [PrototypeTask(task_id="a", question="Q"), PrototypeTask(task_id="b", question="Q")]
    result, _ = run(tmp_path, tasks=tasks, responder=interrupt)
    assert result["exit_code"] == 130
    rows = prototype_artifacts.read_jsonl(tmp_path / "run" / "results.jsonl")
    assert [r["outcome"] for r in rows] == ["agreement", "cancelled"]
    events = prototype_artifacts.read_jsonl(tmp_path / "run" / "trace.jsonl")
    assert any(e["type"] == "contribution" and e["trial_id"] == rows[1]["trial_id"] for e in events)
    with pytest.raises(FileExistsError):
        run(tmp_path)


def test_gpu_requirement_blocks_scripted_run(tmp_path):
    result, _ = run(tmp_path, require_gpu=True)
    assert result["exit_code"] == 1 and not result["results"]
    assert result["manifest"]["status"] == "preflight_failed"


def test_shared_falsey_ingestion_empty_evaluator_and_imported_executor():
    from benchmarks.evaluator import BenchmarkEvaluator
    from contracts.schemas import BenchmarkTask, BenchmarkType
    from ingest.kramabench import KramaBenchAdapter
    from ingest.medagentbench import MedAgentBenchAdapter
    from benchmarks.mscore import MSCoReIngestor, BatchTrialRunner
    task = BenchmarkTask(task_id="x", benchmark_type=BenchmarkType.MSCORE, problem_statement="Q", ground_truth="4")
    assert not BenchmarkEvaluator.evaluate(task, "").normalized_match
    assert KramaBenchAdapter().load_from_dict({"id": "x", "question": "Q", "expected_output": 0}).ground_truth == "0"
    assert MedAgentBenchAdapter().load_from_dict({"id": "x", "question": "Q", "sol": False}).ground_truth == "False"
    assert MSCoReIngestor._parse_raw_dict({"id": "x", "question": "Q", "answer": 0}, "x").ground_truth == "0"
    config = BatchTrialRunner().generate_ablation_matrix(MSCoReIngestor.get_sample_dataset()[:1])[0]
    baseline = asyncio.run(BatchTrialRunner(streaming_enabled=False).run_single_trial(config))
    assert asyncio.run(BatchTrialRunner(streaming_enabled=False, trial_executor=lambda _: baseline).run_single_trial(config))


def test_viewer_escapes_task_markup_and_tolerates_partial_jsonl(tmp_path):
    state = {"manifest": {"tasks": [{"question": "</script><img src=x onerror=alert(1)>"}]}, "results": [], "events": []}
    html = prototype_artifacts.viewer_html(state)
    assert "</script><img" not in html
    assert "\\u003c/script" in html
    path = tmp_path / "trace.jsonl"
    path.write_text('{"type":"ok"}\n{"type":')
    assert prototype_artifacts.read_jsonl(path) == [{"type": "ok"}]


def test_live_demo_server_reads_new_artifacts_and_blocks_other_paths(tmp_path):
    run(tmp_path)
    root = tmp_path / "run"
    server = make_server(root, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(url + "/") as response:
            assert b"Three perspectives" in response.read()
        with (root / "trace.jsonl").open("a") as f:
            f.write('{"type":"new_live_event"}\n')
        with urllib.request.urlopen(url + "/api/state") as response:
            assert json.load(response)["events"][-1]["type"] == "new_live_event"
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(url + "/../references.json")
        assert exc.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_ollama_http_compute_transport_preflight_and_retry_accounting(tmp_path):
    """Exercise real HTTP code with a fake Ollama server; this is not a GPU test."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from llm_broker.backends.ollama import OllamaBackend
    captured = []
    failed_once = False
    class FakeOllama(BaseHTTPRequestHandler):
        def send(self, data):
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def do_GET(self):
            if self.path == "/api/version": self.send({"version": "fake-test-server"})
            elif self.path == "/api/tags": self.send({"models": [{"name": "test-model", "digest": "test-digest"}]})
            elif self.path == "/api/ps": self.send({"models": [{"name": "test-model", "size": 100, "size_vram": 100}]})
            else: self.send_error(404)
        def do_POST(self):
            nonlocal failed_once
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/api/show":
                self.send({"model_info": {"qwen2.context_length": 32768}}); return
            assert self.path == "/api/chat"
            captured.append(data)
            req = LLMRequest(messages=data["messages"])
            ordinary = "Real unique widget question?" in req.messages[-1].content
            if ordinary and not failed_once:
                failed_once = True
                self.send_error(503, "test transient failure"); return
            self.send({"message": {"content": answer(req) if ordinary else '{"ok":true}'},
                       "prompt_eval_count": 100, "eval_count": 50})
        def log_message(self, *args): pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        settings = InferenceSettings(model="test-model", ollama_host=f"http://127.0.0.1:{server.server_port}")
        broker = ModelBroker(OllamaBackend(settings.ollama_host), "test-model", backoff_s=0)
        result = run_prototype([PrototypeTask(task_id="machines", question="Real unique widget question?")],
            {"machines": 5}, tmp_path / "run", settings=settings, seeds=[42], broker=broker)
        r = result["results"][0]
        assert result["exit_code"] == 0 and r["correct"] and r["answer_agreement"]
        assert result["manifest"]["mode"] == "live" and result["manifest"]["model_digest"] == "test-digest"
        assert r["usage"]["calls"] == 4 and r["usage"]["backend_attempts"] == 5
        assert r["usage"]["known_prompt_tokens"] + r["usage"]["known_completion_tokens"] == 600
        assert r["usage"]["unreported_retry_attempts"] == 1 and r["usage"]["total_tokens"] is None
        assert len(captured) == 6  # preflight + ordinary attempts
        assert all(d["options"]["num_ctx"] == 8192 and not d["stream"] for d in captured)
        assert all("format" in d for d in captured[1:])
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)


def test_bundle_contains_current_source_but_excludes_secrets(tmp_path):
    from scripts.prototype_bundle import bundle
    from zipfile import ZipFile
    import hashlib
    root = tmp_path / "project"
    (root / "src" / "scripts").mkdir(parents=True)
    (root / "pyproject.toml").write_text("metadata")
    (root / "src" / "scripts" / "prototype_run.py").write_text("current code")
    (root / ".env").write_text("SECRET")
    destination = tmp_path / "source.zip"
    manifest = bundle(root, destination)
    with ZipFile(destination) as archive:
        assert ".env" not in archive.namelist()
        assert archive.read("src/scripts/prototype_run.py") == b"current code"
        saved = json.loads(archive.read("prototype_source_manifest.json"))
    assert saved == manifest
    assert saved["files"]["src/scripts/prototype_run.py"] == hashlib.sha256(b"current code").hexdigest()


def test_export_failure_preserves_results_and_manifest(tmp_path, monkeypatch):
    from scripts import prototype_run
    def fail(*args): raise OSError("test export failure")
    monkeypatch.setattr(prototype_run, "export_artifacts", fail)
    result, _ = run(tmp_path)
    assert result["exit_code"] == 1
    assert "test export failure" in result["manifest"]["export_error"]
    assert prototype_artifacts.read_jsonl(tmp_path / "run" / "results.jsonl")
