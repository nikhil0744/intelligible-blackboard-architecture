"""Student 2 tests: broker concurrency/retries, parser robustness, agent lifecycle. Offline (MockBackend)."""

import asyncio
import json
import threading

import pytest

from agents import InMemoryBoard, PEXAgent, StaleTurnError, act_parallel, build_panel
from agents.base import AgentTurnError
from contracts.schemas import AgentContribution, AgentRole, BenchmarkType, PXPTag
from llm_broker import ChatMessage, LLMError, LLMRequest, ModelBroker
from llm_broker.backends.mock import MockBackend, default_decision
from prompts import (
    DecisionParseError,
    decision_json_schema,
    domains,
    get_persona,
    list_personas,
    parse_decision,
    render_board,
)


def req(agent="a"):
    return LLMRequest(messages=[ChatMessage(role="user", content="hi")], agent_id=agent)


def broker_with(backend, **kw):
    kw.setdefault("backoff_s", 0.0)
    return ModelBroker(backend, default_model="mock-7b", **kw)


# ---------------- broker ----------------
def test_broker_respects_concurrency_cap():
    be = MockBackend(delay_s=0.05)
    with broker_with(be, max_concurrency=2, max_workers=8) as b:
        out = b.generate_many([req(str(i)) for i in range(8)])
    assert len(out) == 8
    assert be.max_in_flight == 2


def test_broker_parallel_is_faster_than_serial():
    import time

    be = MockBackend(delay_s=0.1)
    with broker_with(be, max_concurrency=4) as b:
        t0 = time.perf_counter()
        b.generate_many([req() for _ in range(4)])
        assert time.perf_counter() - t0 < 0.3


def test_broker_retries_then_succeeds_and_counts_tokens():
    be = MockBackend(fail_first_n=2)
    with broker_with(be, max_retries=2) as b:
        r = b.generate(req("x"))
        assert r.attempts == 3
        u = b.usage_by_agent()["x"]
        assert u.calls == 1 and u.total_tokens > 0


def test_broker_raises_after_retries():
    with broker_with(MockBackend(fail_first_n=5), max_retries=1) as b:
        with pytest.raises(LLMError):
            b.generate(req("y"))
        assert b.usage().failures == 1


def test_broker_async():
    with broker_with(MockBackend()) as b:
        r = asyncio.run(b.agenerate(req()))
        assert r.backend == "mock"


# ---------------- prompts ----------------
def test_persona_matrix_covers_all_benchmark_domains():
    assert {"general", "medical", "engineering", "data"} <= set(domains())
    for d in domains():  # every domain panel has someone whose job is to disagree
        assert AgentRole.CRITIC in {p.role for p in list_personas(d)}


@pytest.mark.parametrize(
    "raw",
    [
        json.dumps(default_decision()),
        "Sure! Here is my answer:\n```json\n" + json.dumps(default_decision()) + "\n```",
        json.dumps(default_decision())[:-1] + ",}",  # trailing comma
        json.dumps({**default_decision(), "tag": "ratify ", "target_contribution_id": "null"}),
    ],
)
def test_parser_tolerates_common_small_model_output(raw):
    d = parse_decision(raw)
    assert d.tag in PXPTag


def test_parser_fixes_percent_confidence_and_string_explanation():
    raw = json.dumps({"tag": "REVISE", "target_contribution_id": None,
                      "prediction": {"claim": "42", "confidence": 85},
                      "explanation": "because"})
    d = parse_decision(raw)
    assert d.prediction.confidence == 0.85 and d.explanation.rationale == "because"


def test_parser_rejects_bad_tag():
    with pytest.raises(DecisionParseError):
        parse_decision(json.dumps({**default_decision(), "tag": "MAYBE"}))


def test_json_schema_tags_match_contract():
    assert set(decision_json_schema()["properties"]["tag"]["enum"]) == {t.value for t in PXPTag}


# ---------------- agents ----------------
def new_board():
    board = InMemoryBoard()
    snap = board.create_session("t1", "What is 6*7?", {"hint": "arithmetic"})
    return board, snap.session_id


def test_first_turn_is_forced_to_revise_and_valid_contract():
    be = MockBackend([json.dumps(default_decision(tag="REJECT", claim="42", target="bogus"))])
    board, sid = new_board()
    with broker_with(be) as b:
        agent = PEXAgent("a0", "general_reasoner", b)
        c = agent.step(board, sid)
    assert isinstance(c, AgentContribution)
    assert c.tag == PXPTag.REVISE and c.target_contribution_id is None and c.turn_index == 0
    assert c.token_usage > 0 and c.metadata["persona"] == "general_reasoner"
    assert board.get_snapshot(sid).active_claim.claim == "42"


def test_invalid_target_is_repointed_and_ratify_inherits_claim():
    board, sid = new_board()
    with broker_with(MockBackend([json.dumps(default_decision(claim="42"))])) as b:
        first = PEXAgent("a0", "general_reasoner", b).step(board, sid)
    ratify = json.dumps(default_decision(tag="RATIFY", claim="forty two", target="nope"))
    with broker_with(MockBackend([ratify])) as b:
        c = PEXAgent("a1", "general_critic", b).step(board, sid)
    assert c.tag == PXPTag.RATIFY
    assert c.target_contribution_id == first.contribution_id
    assert c.payload.prediction.claim == "42"


def test_agent_repairs_invalid_json_via_reprompt():
    be = MockBackend(["I think the answer is 42.", json.dumps(default_decision(claim="42"))])
    board, sid = new_board()
    with broker_with(be) as b:
        c = PEXAgent("a0", "general_reasoner", b).step(board, sid)
    assert c.metadata["parse_attempts"] == 2
    assert be.calls[1].messages[-1].content.startswith("Your previous output was invalid")


def test_agent_gives_up_after_parse_retries():
    board, sid = new_board()
    with broker_with(MockBackend(["nope"])) as b:
        with pytest.raises(AgentTurnError):
            PEXAgent("a0", "general_reasoner", b, max_parse_retries=1).step(board, sid)


def test_counterfactual_hooks_for_s3():
    be = MockBackend()
    board, sid = new_board()
    with broker_with(be) as b:
        a = PEXAgent("a0", "general_reasoner", b, counterfactual_capable=True)
        a.system_addendum = "Reconsider your earlier REJECT."
        c = a.act(board.get_snapshot(sid), is_counterfactual=True)
    assert c.is_counterfactual and c.agent_role == AgentRole.COUNTERFACTUAL
    assert "Reconsider your earlier REJECT." in be.calls[0].messages[0].content


def test_panel_density_and_domain():
    with broker_with(MockBackend()) as b:
        panel = build_panel(b, domain="medical", size=3, counterfactual_density=0.66)
        assert sum(a.counterfactual_capable for a in panel) == 2
        assert all(a.persona.domain == "medical" for a in panel)
        assert len({a.agent_id for a in panel}) == 3
        assert sum(a.counterfactual_capable for a in build_panel(b, size=3, counterfactual_density=0.33)) == 1
        assert sum(a.counterfactual_capable for a in build_panel(b, size=3, counterfactual_density=0.0)) == 0


def test_act_parallel_same_snapshot_order_preserved():
    board, sid = new_board()
    be = MockBackend(delay_s=0.05)
    with broker_with(be, max_concurrency=3) as b:
        panel = build_panel(b, size=3)
        outs = act_parallel(panel, board.get_snapshot(sid))
    assert [o.agent_id for o in outs] == [a.agent_id for a in panel]
    assert be.max_in_flight == 3


def test_stale_turn_is_rejected_by_stub_board():
    board, sid = new_board()
    with broker_with(MockBackend()) as b:
        a = PEXAgent("a0", "general_reasoner", b)
        snap = board.get_snapshot(sid)
        c1, c2 = a.act(snap), a.act(snap)
        board.submit(c1)
        with pytest.raises(StaleTurnError):
            board.submit(c2)


def test_multi_turn_debate_on_stub_board():
    """3 agents, 6 turns: responder reacts to board content -> REVISE then RATIFY/REFUTE."""

    def responder(r: LLMRequest):
        user = r.messages[1].content
        if "(empty" in user:
            return json.dumps(default_decision(claim="42"))
        tag = "REFUTE" if "general_critic" in r.messages[0].content.lower() and "REFUTE" not in user else "RATIFY"
        return json.dumps(default_decision(tag=tag, claim="42"))

    board, sid = new_board()
    with broker_with(MockBackend(responder=responder)) as b:
        panel = build_panel(b, size=3)
        for t in range(6):
            panel[t % 3].step(board, sid)
    snap = board.get_snapshot(sid)
    assert len(snap.contributions) == 6
    assert [c.turn_index for c in snap.contributions] == list(range(6))
    assert snap.contributions[0].tag == PXPTag.REVISE
    assert "turn 5" in render_board(snap) or "turn 4" in render_board(snap)
