"""Student 2 tests for S2.4 / S2.6: inference settings, preflight, validation artifact. Offline."""

import json

import pytest

from llm_broker import InferenceSettings, ModelBroker, build_broker
from llm_broker.backends.mock import MockBackend, default_decision
from llm_broker.backends.ollama import OllamaBackend
from llm_broker.preflight import format_report, run_preflight
from scripts import s2_smoke

LLM_VARS = ["LLM_BACKEND", "LLM_MODEL", "LLM_TEMPERATURE", "LLM_NUM_CTX", "LLM_MAX_TOKENS", "LLM_MAX_REPAIRS",
            "LLM_TIMEOUT_S", "LLM_MAX_CONCURRENCY", "LLM_MAX_RETRIES", "OLLAMA_HOST", "LLM_API_BASE"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for v in LLM_VARS:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.chdir(tmp_path)  # no stray .env, and default outputs/ lands in a temp dir


# ---------------- settings ----------------
def test_pilot_defaults_match_the_agreed_configuration():
    s = InferenceSettings.from_env()
    assert (s.model, s.temperature, s.num_ctx, s.max_tokens, s.max_repairs, s.timeout_s, s.max_concurrency) == (
        "qwen2.5:7b-instruct-q4_K_M", 0.3, 8192, 768, 2, 120.0, 1)
    assert s.agent_kwargs() == {"temperature": 0.3, "max_tokens": 768, "max_parse_retries": 2, "context_tokens": 8192}


def test_settings_read_env_and_overrides_and_reach_broker_and_backend(monkeypatch):
    monkeypatch.setenv("LLM_NUM_CTX", "4096")
    monkeypatch.setenv("LLM_MAX_TOKENS", "512")
    s = InferenceSettings.from_env(backend="ollama", llm_model="other:7b", model=None)
    assert s.num_ctx == 4096 and s.max_tokens == 512 and s.model == "other:7b"
    b = build_broker(s)
    assert b.settings is s and b.default_model == "other:7b" and b.max_concurrency == 1
    assert b.backend.num_ctx == 4096 and b.backend.timeout_s == 120.0


def test_unknown_backend_is_rejected_not_substituted():
    with pytest.raises(ValueError):
        build_broker(InferenceSettings(backend="openai-paid"))


# ---------------- preflight ----------------
class FakeOllama(OllamaBackend):
    """Ollama backend with canned HTTP answers (no server needed)."""

    def __init__(self, models, context_length=32768, **kw):
        super().__init__(**kw)
        self._models, self._ctx = models, context_length

    def _get(self, path):
        if path == "/api/version":
            return {"version": "0.0-test"}
        if path == "/api/tags":
            return {"models": [{"name": m, "digest": "sha256:" + m, "size": 1} for m in self._models]}
        if path == "/api/ps":
            return {"models": [{"name": m, "size": 10, "size_vram": 10} for m in self._models[:1]]}
        raise AssertionError(path)

    def _post(self, path, body):
        if path == "/api/show":
            return {"model_info": {"qwen2.context_length": self._ctx}}
        return {"message": {"content": '{"ok": true}'}, "prompt_eval_count": 12, "eval_count": 5}


def check(report, name):
    return next(c for c in report["checks"] if c["name"] == name)


def test_preflight_ready_records_digest_settings_and_probe():
    s = InferenceSettings()
    b = ModelBroker(FakeOllama([s.model]), default_model=s.model, backoff_s=0.0)
    r = run_preflight(b, s)
    assert r["ok"] and r["model_digest"] == "sha256:" + s.model
    assert r["settings"]["num_ctx"] == 8192 and r["probe"]["prompt_tokens"] == 12
    assert check(r, "usage_reported")["ok"] and check(r, "model_fully_on_gpu")["ok"]
    assert "READY" in format_report(r) and "dependencies" in r and "hardware" in r


def test_preflight_missing_model_fails_without_substituting_another():
    s = InferenceSettings()
    be = FakeOllama(["llama3.1:8b"])
    b = ModelBroker(be, default_model=s.model, backoff_s=0.0)
    r = run_preflight(b, s)
    assert not r["ok"] and not check(r, "model_available")["ok"]
    assert "ollama pull" in check(r, "model_available")["detail"]
    assert b.default_model == s.model and len(b.ledger) == 0  # no probe call was made with another model


def test_preflight_unavailable_backend_is_reported_not_raised():
    s = InferenceSettings(ollama_host="http://127.0.0.1:9", timeout_s=2)
    r = run_preflight(build_broker(s), s)
    assert not r["ok"] and not check(r, "backend_reachable")["ok"]
    assert "NOT READY" in format_report(r)


def test_preflight_flags_context_window_larger_than_the_model_supports():
    s = InferenceSettings()
    b = ModelBroker(FakeOllama([s.model], context_length=4096), default_model=s.model, backoff_s=0.0)
    r = run_preflight(b, s, probe=False)
    assert not r["ok"] and not check(r, "context_window_supported")["ok"]


def test_ollama_backend_reports_missing_usage_as_unknown():
    class NoCounts(FakeOllama):
        def _post(self, path, body):
            return {"message": {"content": "{}"}}

    from llm_broker import ChatMessage, LLMRequest

    resp = NoCounts(["m"]).complete(LLMRequest(messages=[ChatMessage(role="user", content="hi")]), "m")
    assert resp.prompt_tokens is None and resp.completion_tokens is None and resp.total_tokens is None


# ---------------- validation artifact (S2.6) ----------------
def smoke(tmp_path, *extra):
    out = tmp_path / "artifact"
    result = s2_smoke.run_smoke(s2_smoke.build_parser().parse_args(["--backend", "mock", "--out", str(out), *extra]))
    return result, out


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("extra", [(), ("--scheduler",)])
def test_smoke_writes_a_complete_artifact(tmp_path, extra):
    result, out = smoke(tmp_path, "--turns", "4", "--capable", "2", *extra)
    assert result["exit_code"] == 0
    assert {p.name for p in out.iterdir()} == {
        "manifest.json", "calls.jsonl", "raw_responses.jsonl", "contributions.jsonl", "failures.jsonl", "summary.json"}
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == ("scheduler" if extra else "direct")
    assert manifest["settings"]["max_tokens"] == 768 and manifest["preflight"]["ok"]
    assert sum(a["counterfactual_capable"] for a in manifest["agents"]) == 2
    assert all(a["max_tokens"] == 768 and a["context_tokens"] == 8192 for a in manifest["agents"])
    assert summary["turns_succeeded"] == 4 and summary["turns_failed"] == 0
    calls = jsonl(out / "calls.jsonl")
    assert len(calls) == 4 and {c["trial_id"] for c in calls} == {manifest["run_id"]}  # the probe call is not counted
    assert summary["usage"]["calls"] == 4 and summary["usage"]["total_tokens"] == sum(
        c["prompt_tokens"] + c["completion_tokens"] for c in calls)
    assert len(jsonl(out / "contributions.jsonl")) == 4 and len(jsonl(out / "raw_responses.jsonl")) == 4


def test_smoke_records_repairs_failures_and_raw_responses(tmp_path, monkeypatch):
    good = json.dumps(default_decision(claim="5 cents"))
    script = ["not json", good,          # turn 0: repaired on the 2nd response
              "nope", "nope", "nope"]    # turn 1: never valid -> recorded failure

    def scripted(settings):
        b = ModelBroker(MockBackend(script), default_model=settings.model, backoff_s=0.0)
        b.settings = settings
        return b

    monkeypatch.setattr(s2_smoke, "build_broker", scripted)
    result, out = smoke(tmp_path, "--turns", "2", "--no-probe")
    summary = result["summary"]
    assert result["exit_code"] == 0 and summary["turns_succeeded"] == 1 and summary["turns_failed"] == 1
    assert summary["failure_kinds"] == {"invalid_output": 1}
    assert summary["repair"]["turns_valid_after_repair"] == 1 and summary["repair"]["repair_calls"] == 3
    assert summary["repair"]["rejection_reasons"] == {"invalid_json": 4}
    raw = jsonl(out / "raw_responses.jsonl")
    assert [r["accepted"] for r in raw] == [False, True, False, False, False]
    assert raw[0]["raw"] == "not json" and raw[0]["rejected_because"].startswith("invalid_json")
    (failure,) = jsonl(out / "failures.jsonl")
    assert failure["turn"] == 1 and failure["attempts"] == 3 and len(failure["raw_responses"]) == 3


def test_smoke_stops_before_any_task_when_preflight_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")
    out = tmp_path / "artifact"
    code = s2_smoke.main(["--backend", "ollama", "--out", str(out)])
    assert code == 1
    assert json.loads((out / "manifest.json").read_text(encoding="utf-8"))["preflight"]["ok"] is False
    assert not (out / "summary.json").exists()


def test_smoke_accepts_a_real_task_file(tmp_path):
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"task_id": "mscore-17", "question": "Which stage failed?", "context": {"line": "B"}}), encoding="utf-8")
    result, out = smoke(tmp_path, "--turns", "1", "--task-file", str(task))
    assert result["manifest"]["task"]["task_id"] == "mscore-17"
    assert jsonl(out / "contributions.jsonl")[0]["session_id"]
