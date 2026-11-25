import json
import re

import pytest

from benchmarks.batch import run_batch
from benchmarks.runtime import Limits
from llm_broker import LLMResponse

from .test_benchmark_execution import READINESS, SETTINGS, ScriptedBackend, broker, decision, prepared


class RecoveryBackend(ScriptedBackend):
    """Choose scripted actions from the live board and tool observations."""

    def __init__(self, *, fail_evidence=False, confirm=True):
        super().__init__()
        self.fail_evidence = fail_evidence
        self.confirm = confirm

    def complete(self, request, model):
        system = request.messages[0].content
        user = request.messages[1].content
        in_recovery = '"recovery":' in user
        root = re.search(r"Current claim root id=([^ .]+)", system)
        if root:
            if in_recovery and self.confirm:
                action = decision("final", "RATIFY", 6, root[1])
            else:
                action = decision("contribution", "REFUTE", 5, root[1])
        elif request.json_schema.get("title") == "ToolAction":
            if not in_recovery and self.fail_evidence:
                action = {"kind": "tool", "name": "query", "arguments": {"file": "input.csv",
                          "aggregate": {"column": "missing", "op": "sum"}}}
            elif in_recovery and "prior branch" in system:
                action = {"kind": "tool", "name": "query", "arguments": {"file": "input.csv",
                          "select": ["value"]}}
            else:
                # Changing the ignored limit on an aggregate is not a new investigation.
                action = {"kind": "tool", "name": "query", "arguments": {"file": "input.csv",
                          "aggregate": {"column": "value", "op": "sum"}, "limit": 2 if in_recovery else 1}}
        else:
            assert '"value": 6.0' in user or '"rows": [{"value": "2"}, {"value": "4"}]' in user
            action = decision("final", "REVISE", 6 if in_recovery else 5)
        return LLMResponse(text=json.dumps(action), model=model, backend=self.name)


def run(prepared, tmp_path, backend, limits=None):
    events = []
    store, _ = run_batch(prepared, broker(backend), SETTINGS, tmp_path / "results", limit=1,
                         limits=limits, readiness=READINESS, observer=events.append)
    return store.attempts()[0], events


@pytest.mark.parametrize("fail_evidence", [False, True])
def test_recovery_requires_new_evidence_and_fresh_panel_votes(prepared, tmp_path, fail_evidence):
    result, events = run(prepared, tmp_path, RecoveryBackend(fail_evidence=fail_evidence))
    assert result["status"] == "consensus"
    assert result["recovery"]["attempts"] == 1
    assert result["recovery"]["confirmed"] and result["recovery"]["applied"]
    assert result["recovery"]["calls"] > 0
    assert result["output"] == 6 and result["grade"]["correct"]
    start = next(event for event in events if event["kind"] == "recovery_started")
    assert start["reason"] == ("insufficient_evidence" if fail_evidence else "deadlock")
    if not fail_evidence:
        assert any(event.get("reason") == "prior_branch_query" for event in events)
    candidate = next(event for event in events if event["kind"] == "recovery_candidate")
    observations = {event["request_id"]: event for event in events if event["kind"] == "tool_observed"}
    assert observations[candidate["evidence_request_id"]]["branch"] == "recovery"
    branch_posts = [event["contribution"] for event in events
                    if event["kind"] == "contribution" and event["branch"] == "recovery"]
    assert {post["agent_id"] for post in branch_posts if post["tag"] == "RATIFY"} == {"analyst", "critic", "reasoner"}
    assert all(post["target_contribution_id"] == candidate["contribution_id"]
               for post in branch_posts if post["tag"] == "RATIFY")
    finish = next(event for event in events if event["kind"] == "recovery_finished")
    assert finish["votes"]["ratified_agent_ids"] == ["analyst", "critic", "reasoner"]
    assert finish["votes"]["pending_agent_ids"] == []
    prompts = [event["messages"][1]["content"] for event in events
               if event["kind"] == "model_response" and event["branch"] == "recovery"]
    assert any('"ratified_agent_ids": ["critic"]' in prompt for prompt in prompts)
    assert any('"pending_agent_ids": ["analyst"]' in prompt for prompt in prompts)
    assert 1 < result["tool_rounds"] <= Limits().max_tool_rounds


def test_unconfirmed_recovery_preserves_main_submission(prepared, tmp_path):
    result, events = run(prepared, tmp_path, RecoveryBackend(confirm=False))
    assert result["status"] == "deadlock" and result["output"] == 5
    assert result["recovery"]["attempts"] == 1
    assert not result["recovery"]["applied"] and not result["recovery"]["confirmed"]
    assert not result["grade"]["correct"]
    assert any(event["kind"] == "recovery_candidate" for event in events)


def test_recovery_respects_shared_tool_budget(prepared, tmp_path):
    result, events = run(prepared, tmp_path, RecoveryBackend(), Limits(max_tool_rounds=1))
    assert result["status"] == "deadlock" and result["tool_rounds"] == 1
    assert result["recovery"]["attempts"] == 0
    assert any(event["kind"] == "recovery_skipped" for event in events)


def test_recovery_opt_out_preserves_baseline(prepared, tmp_path):
    result, events = run(prepared, tmp_path, RecoveryBackend(), Limits(recovery_enabled=False))
    assert result["recovery"]["mode"] == "disabled"
    assert result["recovery"]["attempts"] == 0 and result["status"] == "deadlock"
    assert all(event.get("branch", "main") == "main" for event in events)
