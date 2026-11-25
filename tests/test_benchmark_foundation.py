import json
import re

import pytest

from agents import InMemoryBoard, build_panel, register_panel
from contracts.schemas import BlackboardStatus
from llm_broker import InferenceSettings, ModelBroker
from llm_broker.backends.mock import MockBackend, default_decision
from scheduler.engine import DeterministicScheduler
from scheduler.policies import RoundRobinPolicy


def test_real_agent_classes_run_through_canonical_scheduler():
    def respond(request):
        ids = re.findall(r"\[id=([^\]]+)\]", request.messages[-1].content)
        return json.dumps(default_decision("RATIFY" if ids else "REVISE", "answer", ids[-1] if ids else None))
    board = InMemoryBoard()
    board.create_session("task", "Actual task question", session_id="session")
    with ModelBroker(MockBackend(responder=respond), "mock") as broker:
        panel = build_panel(broker, size=3, counterfactual_density=2/3)
        scheduler = DeterministicScheduler("session", board, policy=RoundRobinPolicy())
        register_panel(scheduler, panel)
        snapshot = scheduler.run(6)
    assert snapshot.status == BlackboardStatus.CONSENSUS
    assert [c.turn_index for c in snapshot.contributions] == list(range(3))
    assert len({c.agent_id for c in snapshot.contributions}) == 3
    assert all(c.agent_role.value != "COUNTERFACTUAL" for c in snapshot.contributions)


def test_inference_settings_do_not_accept_invalid_resource_limits():
    for kwargs in ({"timeout_s": 0}, {"max_tokens": 8192}, {"max_retries": -1}):
        with pytest.raises(ValueError):
            InferenceSettings(**kwargs)
