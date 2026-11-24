"""Regression cases for the model-facing benchmark and Ollama paths."""

import http.client
import json

import pytest

from agents.board_client import InMemoryBoard
from benchmarks.actions import BenchmarkAgent, ToolAction
from llm_broker import ChatMessage, LLMError, LLMRequest, ModelBroker
from llm_broker.backends.mock import MockBackend
from llm_broker.backends.ollama import OllamaBackend
from prompts import DecisionParseError


def final(claim, output, *, tag="REVISE", target=None):
    return json.dumps({
        "kind": "final", "tag": tag, "target_contribution_id": target,
        "prediction": {"claim": claim, "confidence": 0.7},
        "explanation": {"rationale": "I checked the result against the tool observation."},
        "output": output,
    })


def test_opening_allows_query_refinement_or_a_supported_submission():
    query = json.dumps({
        "kind": "tool", "name": "query",
        "arguments": {"file": "reports.csv", "filters": [{"column": "Year", "op": "ge", "value": 2022}]},
    })
    claim = "The total number of fraud reports by FTC over the web between 2022 and 2024 is 17,338,277."
    backend = MockBackend([query, final(claim, "17,338,277")])
    with ModelBroker(backend, default_model="mock") as broker:
        agent = BenchmarkAgent("analyst", "data_analyst", broker, max_parse_retries=0)
        snapshot = InMemoryBoard().create_session("task", "How many reports from 2022 through 2024?")
        action = agent.action(snapshot, phase="opening")
        assert isinstance(action, ToolAction) and action.arguments["filters"][0]["value"] == 2022
        assert "more specific query" in backend.calls[0].messages[0].content
        contribution = agent.action(snapshot, phase="submit")
        assert contribution.payload.prediction.claim == claim
        assert contribution.metadata["final_output"] == "17338277"


@pytest.mark.parametrize("claim", [
    "The answer is 4 or 5.",
    "The answer is 4, but the answer is 5.",
    "The count for 2024 is 5, while the overall answer is 4.",
    "The answer is 4 to 5.",
    "The answer is 5.",
])
def test_final_rejects_ambiguous_or_mismatched_claims(claim):
    agent = BenchmarkAgent("analyst", "data_analyst", None)
    with pytest.raises(DecisionParseError):
        agent.parse_output(final(claim, 4))


def test_ratify_accepts_numeric_equivalence_and_preserves_target():
    backend = MockBackend([
        final("For 2024 the total is 1,240.", 1240),
        final("1240.0", 1240, tag="RATIFY"),
    ])
    board = InMemoryBoard()
    initial = board.create_session("task", "How many reports in 2024?")
    with ModelBroker(backend, default_model="mock") as broker:
        agent = BenchmarkAgent("analyst", "data_analyst", broker, max_parse_retries=0)
        opening = agent.action(initial, phase="submit")
        snapshot = board.submit(opening)
        backend.responses[1] = final("1240.0", 1240, tag="RATIFY", target=opening.contribution_id)
        contribution = agent.action(snapshot, phase="review_closed")
        assert contribution.target_contribution_id == opening.contribution_id
        assert contribution.payload.prediction.claim == opening.payload.prediction.claim


def test_model_repairs_keep_only_latest_error_and_phase_constraints():
    backend = MockBackend(["invalid-first", "invalid-second", final("4", 4)])
    with ModelBroker(backend, default_model="mock") as broker:
        agent = BenchmarkAgent("analyst", "data_analyst", broker, max_parse_retries=2)
        snapshot = InMemoryBoard().create_session("task", "How many reports in 2024?")
        contribution = agent.action(snapshot, phase="submit")
    assert contribution.metadata["parse_attempts"] == 3
    assert "Tools are closed" in backend.calls[2].messages[0].content
    assert [message.content for message in backend.calls[2].messages if message.role == "assistant"] == ["invalid-second"]
    assert contribution.metadata["raw_responses"][:2] == ["invalid-first", "invalid-second"]


def test_ollama_wraps_incomplete_http_payload(monkeypatch):
    def incomplete(*args, **kwargs):
        raise http.client.IncompleteRead(b'{"message":', 24)

    monkeypatch.setattr("urllib.request.urlopen", incomplete)
    backend = OllamaBackend()
    request = LLMRequest(messages=[ChatMessage(role="user", content="count")])
    with pytest.raises(LLMError, match="Ollama unreachable"):
        backend.complete(request, "mock")


def test_ollama_reports_malformed_usage_as_unknown(monkeypatch):
    backend = OllamaBackend()
    monkeypatch.setattr(backend, "_post", lambda *args: {
        "message": {"content": "answer"}, "prompt_eval_count": float("inf"), "eval_count": 2,
    })
    request = LLMRequest(messages=[ChatMessage(role="user", content="count")])
    result = backend.complete(request, "mock")
    assert result.text == "answer" and result.total_tokens is None
