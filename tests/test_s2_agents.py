"""Student 2 tests: broker concurrency/retries, parser robustness, agent lifecycle. Offline (MockBackend)."""

import asyncio
import json
import re
import threading

import pytest

from agents import InMemoryBoard, PEXAgent, StaleTurnError, act_parallel, build_panel, panel_manifest, register_panel
from agents.base import AgentTurnError
from contracts.schemas import AgentContribution, AgentRole, BenchmarkType, Explanation, PEXPayload, Prediction, PXPTag
from llm_broker import ChatMessage, LLMError, LLMRequest, ModelBroker, usage_scope
from llm_broker.backends.mock import MockBackend, default_decision
from prompts import (
    DecisionParseError,
    build_prompt,
    decision_json_schema,
    domains,
    estimate_tokens,
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


def post_first(board, sid, claim="42"):
    with broker_with(MockBackend([json.dumps(default_decision(claim=claim))])) as b:
        return PEXAgent("a0", "general_reasoner", b).step(board, sid)


def test_first_turn_revise_is_valid_contract_and_drops_impossible_target():
    be = MockBackend([json.dumps(default_decision(tag="REVISE", claim="42", target="bogus"))])
    board, sid = new_board()
    with broker_with(be) as b:
        c = PEXAgent("a0", "general_reasoner", b).step(board, sid)
    assert isinstance(c, AgentContribution)
    assert c.tag == PXPTag.REVISE and c.target_contribution_id is None and c.turn_index == 0
    assert c.token_usage > 0 and c.metadata["persona"] == "general_reasoner"
    assert c.metadata["normalizations"] == ["dropped_target_on_empty_board"]
    assert board.get_snapshot(sid).active_claim.claim == "42"


def test_non_revise_on_empty_board_is_reprompted_not_rewritten():
    be = MockBackend([
        json.dumps(default_decision(tag="REJECT", claim="42", target="bogus")),
        json.dumps(default_decision(tag="REVISE", claim="42")),
    ])
    board, sid = new_board()
    with broker_with(be) as b:
        c = PEXAgent("a0", "general_reasoner", b).step(board, sid)
    assert c.tag == PXPTag.REVISE and c.metadata["parse_attempts"] == 2
    assert c.metadata["repair_reasons"][0].startswith("non_revise_on_empty_board")
    assert len(c.metadata["raw_responses"]) == 2


def test_contradictory_ratify_never_becomes_agreement():
    """S2.2: a RATIFY carrying a different claim must not be rewritten into the target's claim."""
    board, sid = new_board()
    first = post_first(board, sid)
    bad = json.dumps(default_decision(tag="RATIFY", claim="forty three", target=first.contribution_id))
    be = MockBackend([bad])
    with broker_with(be) as b:
        with pytest.raises(AgentTurnError) as ei:
            PEXAgent("a1", "general_critic", b, max_parse_retries=2).step(board, sid)
    e = ei.value
    assert e.kind == "invalid_output" and e.attempts == 3 and e.tokens > 0
    assert len(e.raw_responses) == 3 and all(r == bad for r in e.raw_responses)
    assert all(r.startswith("ratify_claim_mismatch") for r in e.repair_reasons)
    assert e.to_record()["agent_id"] == "a1"
    assert len(board.get_snapshot(sid).contributions) == 1  # nothing was committed
    # the repair prompt carried the real semantic error, not a generic one
    assert "forty three" in be.calls[1].messages[-1].content


def test_contradictory_ratify_can_be_corrected_by_the_model_itself():
    board, sid = new_board()
    first = post_first(board, sid)
    be = MockBackend([
        json.dumps(default_decision(tag="RATIFY", claim="forty three", target=first.contribution_id)),
        json.dumps(default_decision(tag="REVISE", claim="43", target=first.contribution_id)),
    ])
    with broker_with(be) as b:
        c = PEXAgent("a1", "general_critic", b).step(board, sid)
    assert c.tag == PXPTag.REVISE and c.payload.prediction.claim == "43"
    assert c.metadata["parse_attempts"] == 2


def test_ratify_matches_claim_up_to_case_whitespace_punctuation():
    board, sid = new_board()
    first = post_first(board, sid, claim="The answer is 42")
    ok = json.dumps(default_decision(tag="RATIFY", claim="  the answer  is 42. ", target=first.contribution_id))
    with broker_with(MockBackend([ok])) as b:
        c = PEXAgent("a1", "general_critic", b).step(board, sid)
    assert c.tag == PXPTag.RATIFY and c.payload.prediction.claim == "The answer is 42"
    assert c.metadata["parse_attempts"] == 1 and c.metadata["repair_reasons"] == []


@pytest.mark.parametrize("tag,target,code", [
    ("RATIFY", "nope", "unknown_target"),
    ("REFUTE", None, "missing_target"),
    ("REVISE", "nope", "unknown_target"),
])
def test_invalid_target_is_not_redirected(tag, target, code):
    """S2.2: an unknown or missing target must not be pointed at some other post."""
    board, sid = new_board()
    post_first(board, sid)
    with broker_with(MockBackend([json.dumps(default_decision(tag=tag, claim="42", target=target))])) as b:
        with pytest.raises(AgentTurnError) as ei:
            PEXAgent("a1", "general_critic", b, max_parse_retries=1).step(board, sid)
    assert ei.value.repair_reasons[-1].startswith(code)


def test_truncated_id_is_completed_only_when_unique():
    board, sid = new_board()
    first = post_first(board, sid)
    short = json.dumps(default_decision(tag="REFUTE", claim="41", target=first.contribution_id[:8]))
    with broker_with(MockBackend([short])) as b:
        c = PEXAgent("a1", "general_critic", b).step(board, sid)
    assert c.target_contribution_id == first.contribution_id
    assert c.metadata["normalizations"] == ["completed_unique_id_prefix"]


def test_model_failure_is_recorded_as_such():
    board, sid = new_board()
    with broker_with(MockBackend(fail_first_n=9), max_retries=0) as b:
        with pytest.raises(AgentTurnError) as ei:
            PEXAgent("a0", "general_reasoner", b).step(board, sid)
    assert ei.value.kind == "model_failure" and ei.value.raw_responses == []


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
    # capability is recorded separately; the persona keeps its domain role (S2.1)
    assert c.is_counterfactual and c.agent_role == a.persona.role != AgentRole.COUNTERFACTUAL
    assert c.metadata["counterfactual_capable"] is True
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
        target = re.findall(r"\[id=([^\]]+)\]", user)[-1]
        return json.dumps(default_decision(tag=tag, claim="42", target=target))

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


# ---------------- S2.1: scheduler integration + matched conditions ----------------
def debate_responder(r: LLMRequest):
    """First post proposes 42; everyone after ratifies the latest post on the board."""
    user = r.messages[1].content
    if "(empty" in user:
        return json.dumps(default_decision(claim="42"))
    target = re.findall(r"\[id=([^\]]+)\]", user)[-1]
    return json.dumps(default_decision(tag="RATIFY", claim="42", target=target))


def test_all_density_conditions_share_one_roster():
    """Only counterfactual capability may differ between the 0/1/2/3 conditions."""
    with broker_with(MockBackend()) as b:
        specs = [panel_manifest(build_panel(b, domain="medical", size=3, counterfactual_count=n, seed=7))
                 for n in range(4)]
    capable = [{s["agent_id"] for s in sp if s["counterfactual_capable"]} for sp in specs]
    assert [len(c) for c in capable] == [0, 1, 2, 3]
    assert capable[0] < capable[1] < capable[2] < capable[3]  # nested conditions
    strip = lambda sp: [{k: v for k, v in s.items() if k != "counterfactual_capable"} for s in sp]
    assert strip(specs[0]) == strip(specs[1]) == strip(specs[2]) == strip(specs[3])
    assert all(s["role"] != AgentRole.COUNTERFACTUAL.value for sp in specs for s in sp)


def test_capability_does_not_change_the_prompt():
    board, sid = new_board()
    snap = board.get_snapshot(sid)
    prompts_by_n = []
    for n in (0, 3):
        be = MockBackend()
        with broker_with(be) as b:
            act_parallel(build_panel(b, size=3, counterfactual_count=n), snap)
        prompts_by_n.append(sorted((c.agent_id, c.seed, c.temperature, tuple(m.content for m in c.messages))
                                   for c in be.calls))
    assert prompts_by_n[0] == prompts_by_n[1]


def test_counterfactual_count_out_of_range_is_rejected():
    with broker_with(MockBackend()) as b:
        with pytest.raises(ValueError):
            build_panel(b, size=3, counterfactual_count=4)


def test_agent_as_handler_generates_without_committing():
    board, sid = new_board()
    with broker_with(MockBackend()) as b:
        agent = PEXAgent("a0", "general_reasoner", b)
        c = agent(board.get_snapshot(sid), None)
    assert isinstance(c, AgentContribution)
    assert board.get_snapshot(sid).contributions == []


def test_handler_passes_scheduled_target_as_hint_only_if_on_board():
    from contracts.schemas import ScheduledTurn

    board, sid = new_board()
    first = post_first(board, sid)
    snap = board.get_snapshot(sid)
    be = MockBackend(responder=debate_responder)
    with broker_with(be) as b:
        agent = PEXAgent("a1", "general_critic", b)
        agent(snap, ScheduledTurn(session_id=sid, agent_id="a1", target_contribution_id=first.contribution_id))
        agent(snap, ScheduledTurn(session_id=sid, agent_id="a1", target_contribution_id="not-on-board"))
    assert first.contribution_id in be.calls[0].messages[0].content
    assert "not-on-board" not in be.calls[1].messages[0].content


@pytest.mark.parametrize("n_capable", [0, 1, 2, 3])
def test_real_pexagent_runs_under_scheduler_in_every_condition(n_capable):
    """The real PEXAgent class (offline broker) completes a scheduler session; the scheduler commits."""
    from scheduler.engine import DeterministicScheduler
    from scheduler.policies import RoundRobinPolicy
    from storage.redis_store import InMemoryRedisClient, RedisBlackboardStore

    store = RedisBlackboardStore(redis_client=InMemoryRedisClient())
    sid = f"s2_sched_{n_capable}"
    store.create_session(sid, "t1", "What is 6*7?")
    with broker_with(MockBackend(responder=debate_responder)) as b:
        panel = build_panel(b, size=3, counterfactual_count=n_capable)
        # round-robin: the default reactive policy currently starves the third agent (S1.3)
        sched = DeterministicScheduler(session_id=sid, store=store, policy=RoundRobinPolicy(), consensus_threshold=3)
        register_panel(sched, panel)
        snap = sched.run(max_turns=10)
    assert snap.status.value == "CONSENSUS"
    assert {c.agent_id for c in snap.contributions} == {a.agent_id for a in panel}  # capable agents get ordinary turns
    assert all(c.metadata["model"] == "mock-7b" for c in snap.contributions)
    assert sum(1 for c in snap.contributions if c.tag == PXPTag.RATIFY) >= 3


def test_failed_turn_under_scheduler_is_an_error_not_a_ratify():
    from scheduler.engine import DeterministicScheduler
    from storage.redis_store import InMemoryRedisClient, RedisBlackboardStore

    store = RedisBlackboardStore(redis_client=InMemoryRedisClient())
    store.create_session("s2_fail", "t1", "What is 6*7?")
    with broker_with(MockBackend(["not json"])) as b:
        sched = DeterministicScheduler(session_id="s2_fail", store=store)
        register_panel(sched, build_panel(b, size=3, max_parse_retries=1))
        with pytest.raises(AgentTurnError):
            sched.step()
    assert store.get_snapshot("s2_fail").contributions == []


# ---------------- S2.5: usage ledger, unknown usage, broker lifetime, prompt budget ----------------
class NoUsageBackend(MockBackend):
    """A backend that does not report token counts."""

    def complete(self, request, model):
        r = super().complete(request, model)
        r.prompt_tokens = r.completion_tokens = None
        return r


def run_scripted_trial(broker, trial_id, size=3):
    board, sid = new_board()
    with usage_scope(trial_id=trial_id):
        for agent in build_panel(broker, size=size):
            agent.step(board, sid)
    return broker.trial_usage(trial_id)


def test_unknown_usage_is_unknown_not_zero():
    board, sid = new_board()
    with broker_with(NoUsageBackend()) as b:
        with usage_scope(trial_id="t-unknown"):
            c = PEXAgent("a0", "general_reasoner", b).step(board, sid)
        u = b.trial_usage("t-unknown")
        assert c.token_usage is None
        assert u.calls == 1 and u.unknown_usage_calls == 1
        assert u.total_tokens is None and u.known_total_tokens == 0
        assert b.usage().unknown_usage_calls == 1


def test_equal_scripted_trials_report_equal_cost_not_cumulative():
    with broker_with(MockBackend(responder=debate_responder)) as b:
        first = run_scripted_trial(b, "trial-1")
        second = run_scripted_trial(b, "trial-2")
        assert first.calls == second.calls == 3
        assert first.total_tokens == second.total_tokens and first.total_tokens > 0
        assert b.usage().total_tokens == first.total_tokens + second.total_tokens  # running total is batch-wide
        assert {r.trial_id for r in b.ledger.records()} == {"trial-1", "trial-2"}


def test_broker_stays_usable_for_later_trials_after_close():
    b = broker_with(MockBackend())
    with b:
        run_scripted_trial(b, "trial-1")
    later = run_scripted_trial(b, "trial-2")  # after __exit__/close()
    assert later.calls == 3 and later.failures == 0
    assert len(b.generate_many([req("x"), req("y")])) == 2  # the thread pool comes back on demand
    assert b.trial_usage("trial-1").calls == 3  # the ledger survived close()
    b.close()


def test_ledger_marks_repairs_phase_model_and_failures():
    board, sid = new_board()
    be = MockBackend(["not json", json.dumps(default_decision(claim="42"))])
    with broker_with(be) as b:
        with usage_scope(trial_id="t1", phase="candidate_replay"):
            PEXAgent("a0", "general_reasoner", b).step(board, sid)
        recs = b.ledger.records(trial_id="t1")
    assert [r.is_repair for r in recs] == [False, True]
    assert all(r.phase == "candidate_replay" and r.model == "mock-7b" and r.backend == "mock" for r in recs)
    assert all(r.session_id == sid and r.agent_id == "a0" and r.estimated_prompt_tokens > 0 for r in recs)
    assert b.trial_usage("t1").repairs == 1
    assert set(b.ledger.by_phase("t1")) == {"candidate_replay"}

    with broker_with(MockBackend(fail_first_n=9), max_retries=1) as b:
        with usage_scope(trial_id="t2"):
            with pytest.raises(LLMError):
                b.generate(req("z"))
        (rec,) = b.ledger.records(trial_id="t2")
    assert rec.ok is False and rec.attempts == 2 and rec.error and rec.prompt_tokens is None
    assert rec.phase == "deliberation"  # the default phase


def test_usage_scope_reaches_worker_threads_and_rejects_unknown_phase():
    board, sid = new_board()
    with broker_with(MockBackend(), max_concurrency=3) as b:
        with usage_scope(trial_id="par", phase="baseline_replay"):
            act_parallel(build_panel(b, size=3), board.get_snapshot(sid))
            b.generate_many([req("m1"), req("m2")])
        recs = b.ledger.records(trial_id="par")
    assert len(recs) == 5 and all(r.phase == "baseline_replay" for r in recs)
    with pytest.raises(ValueError):
        with usage_scope(phase="made_up"):
            pass


def test_ledger_can_mirror_to_jsonl(tmp_path):
    path = tmp_path / "run" / "calls.jsonl"
    with broker_with(MockBackend()) as b:
        b.ledger.attach_file(path)
        with usage_scope(trial_id="t1"):
            b.generate(req("a"))
            b.generate(req("b"))
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["agent_id"] for r in rows] == ["a", "b"] and all(r["trial_id"] == "t1" for r in rows)


def long_board(n_posts=30, context=None, rationale_len=300):
    """Board: post 0 = REVISE '42'; then alternating REFUTEs; every post targets the previous one."""
    board = InMemoryBoard()
    sid = board.create_session("t-long", "What is 6*7? Explain.", context or {}).session_id
    prev = None
    for i in range(n_posts):
        c = AgentContribution(
            session_id=sid,
            turn_index=i,
            agent_id=f"a{i % 3}",
            agent_role=AgentRole.PRIMARY,
            tag=PXPTag.REVISE if i == 0 else PXPTag.REFUTE,
            target_contribution_id=prev,
            payload=PEXPayload(
                prediction=Prediction(claim="42" if i == 0 else f"not-{i}", confidence=0.5),
                explanation=Explanation(rationale=f"reason {i} " + "x" * rationale_len, evidence=[]),
            ),
        )
        board.submit(c)
        prev = c.contribution_id
    return board.get_snapshot(sid)


def test_prompt_budget_keeps_task_active_position_target_and_dependency():
    snap = long_board(30)
    ids = [c.contribution_id for c in snap.contributions]
    active, target, dependency = ids[0], ids[10], ids[9]  # active = the only REVISE; ids[10] targets ids[9]
    persona = get_persona("general_reasoner")
    msgs, rep = build_prompt(persona, snap, "a0", max_history=30, token_budget=2000, must_keep_ids=[target])
    user = msgs[1].content
    assert rep.estimated_prompt_tokens <= 2000 and not rep.over_budget
    assert "What is 6*7? Explain." in user
    for must in (active, target, dependency):
        assert f"[id={must}]" in user and must in rep.protected_contribution_ids
    assert f"[id={ids[-1]}]" in user  # newest post kept first among the rest
    assert rep.omitted_contribution_ids and rep.truncated
    assert all(f"[id={i}]" not in user for i in rep.omitted_contribution_ids)
    assert rep.posts_included + len(rep.omitted_contribution_ids) == rep.posts_on_board == 30
    assert sum(estimate_tokens(m.content) for m in msgs) <= 2000


def test_prompt_budget_no_truncation_when_it_fits_and_context_is_last_resort():
    persona = get_persona("general_reasoner")
    _, rep = build_prompt(persona, long_board(3), "a0", token_budget=7000)
    assert not rep.truncated and rep.posts_included == 3

    snap = long_board(3, context={"notes": "y" * 5000})
    _, rep = build_prompt(persona, snap, "a0", token_budget=1500)
    assert rep.context_chars_kept < rep.context_chars_total  # context shortened, and recorded
    assert set(rep.protected_contribution_ids) <= {c.contribution_id for c in snap.contributions}
    assert not rep.over_budget

    _, rep = build_prompt(persona, long_board(3), "a0", token_budget=50)
    assert rep.over_budget  # reported, never silently dropped


def test_agent_records_prompt_budget_in_contribution_metadata():
    snap = long_board(30)
    be = MockBackend([json.dumps(default_decision(tag="REFUTE", claim="41", target=snap.contributions[-1].contribution_id))])
    with broker_with(be) as b:
        agent = PEXAgent("a9", "general_critic", b, context_tokens=2400, max_tokens=768, max_history=30)
        assert agent.prompt_token_budget == 2400 - 768 - 64
        c = agent.act(snap)
    pb = c.metadata["prompt_budget"]
    assert pb["token_budget"] == 1568 and pb["omitted_contribution_ids"] and not pb["over_budget"]
    assert pb["estimated_prompt_tokens"] <= 1568
    assert agent.spec()["prompt_token_budget"] == 1568
