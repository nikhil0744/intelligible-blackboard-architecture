"""Observed conflicts, isolated branches, explicit confirmation and honest failures."""
import itertools
import json
from fractions import Fraction as F
from pathlib import Path
import re

import pytest

from llm_broker import InferenceSettings, ModelBroker
from llm_broker.backends.mock import MockBackend
from llm_broker.ledger import current_scope
from scripts import prototype_artifacts
from scripts.prototype_recovery import RESOLVER, run_recovery
from scripts.prototype_run import PrototypeTask, load_inputs


@pytest.fixture(autouse=True)
def no_repeated_plotting(monkeypatch):
    monkeypatch.setattr(prototype_artifacts, "plot_results", lambda *args: None)


def decision(tag, claim, target=None):
    expression = "(1/3) / ((9/10)*(1/3) + (1/3))" if claim == "10/19" else claim
    return json.dumps({"explanation": {"rationale": "Likelihood calculation and an explicit observation check.", "evidence": [f"CALC: {expression} = {claim}"]},
        "prediction": {"claim": claim, "confidence": .7}, "tag": tag, "target_contribution_id": target})


def debate(*, deny_sandbox=False, deny_live=False, wrong_candidate=False, interrupt=False, revise_conflict=False):
    entered = False
    def respond(req):
        nonlocal entered
        if req.agent_id == "_preflight": return '{"ok":true}'
        active = re.search(r"The active proposal is id=([^,]+), claim=", req.messages[0].content)
        target = active[1] if active else None
        phase = current_scope()["phase"]
        if phase == "independent_verification":
            return decision("REVISE", "2/3" if wrong_candidate else "10/19")
        if req.agent_id == RESOLVER:
            if interrupt: raise KeyboardInterrupt
            entered = True
            return decision("REVISE", "2/3" if wrong_candidate else "10/19", target)
        if not entered:
            return decision("REVISE" if not active or revise_conflict else "REFUTE", "2/3" if not active else "10/19", target)
        if (deny_sandbox and phase == "candidate_replay") or (deny_live and phase == "deliberation"):
            return decision("REFUTE", "2/3", target)
        return decision("RATIFY", "2/3" if wrong_candidate else "10/19", target)
    return respond


def run(tmp_path, responder, *, refs=None, **kwargs):
    broker = ModelBroker(MockBackend(responder=responder), "test-model", max_retries=0)
    task = PrototypeTask(task_id="host", question="Biased-host probability question")
    result = run_recovery(task, tmp_path / "run", settings=InferenceSettings(backend="mock", model="test-model"),
                          broker=broker, references=refs or {"host": "10/19"}, **kwargs)
    rows = prototype_artifacts.read_jsonl(tmp_path / "run" / "trace.jsonl")
    return result, broker, rows


@pytest.mark.parametrize("revise", [False, True])
def test_observed_conflict_is_replayed_checked_and_confirmed(tmp_path, revise):
    result, broker, rows = run(tmp_path, debate(revise_conflict=revise))
    assert result["exit_code"] == 0 and result["report"]["recovery_confirmed"]
    assert result["report"]["trigger_observed"] == "first_agent_disagreement"
    assert result["report"]["sandbox_isolated"]
    assert result["report"]["main_hash_before_sandbox"] == result["report"]["main_hash_after_sandbox"]
    posts = [e for e in rows if e["type"] == "contribution"]
    assert len([e for e in posts if e["phase"] == "ordinary_discussion"]) == 2
    cf = [e for e in posts if e["contribution"]["agent_id"] == RESOLVER]
    assert len(cf) == 2 and all(e["contribution"]["is_counterfactual"] for e in cf)
    assert cf[0]["contribution"]["session_id"] != cf[1]["contribution"]["session_id"]
    assert cf[1]["contribution"]["metadata"]["sandbox_source_id"] == cf[0]["contribution"]["contribution_id"]
    assert cf[0]["agreeing_agents"] == [] and cf[1]["agreeing_agents"] == []
    ordinary_and_live = [e["contribution"] for e in posts if e["phase"] in {"ordinary_discussion", "live_promotion", "live_confirmation"}]
    assert [c["turn_index"] for c in ordinary_and_live] == list(range(6))
    saved = prototype_artifacts.read_jsonl(tmp_path / "run" / "results.jsonl")[0]
    calls = broker.ledger.records(trial_id=saved["trial_id"])
    assert len(calls) == 15 and saved["turns_attempted"] == 15
    assert result["report"]["independent_checks"] == 6
    assert saved["correct"] and saved["answer_agreement"] and saved["recovery_enabled"]
    assert saved["usage"]["total_tokens"] == sum(c.prompt_tokens + c.completion_tokens for c in calls)
    assert {c.phase for c in calls} == {"deliberation", "candidate_replay", "independent_verification"}
    assert all(RESOLVER not in e["agreeing_agents"] for e in posts)


def test_immediate_agreement_does_not_invent_conflict_or_call_resolver(tmp_path):
    def agree(req):
        active = re.search(r"The active proposal is id=([^,]+), claim=", req.messages[0].content)
        return decision("RATIFY" if active else "REVISE", "10/19", active[1] if active else None)
    result, broker, rows = run(tmp_path, agree)
    assert result["exit_code"] == 0 and not result["report"]["triggered"]
    assert not any(c.agent_id == RESOLVER for c in broker.backend.calls)
    assert not any(e["type"] == "recovery_stage" for e in rows)
    saved = prototype_artifacts.read_jsonl(tmp_path / "run" / "results.jsonl")[0]
    assert saved["outcome"] == "ordinary_agreement" and saved["correct"]


def test_strict_stall_trigger_waits_for_three_conflicting_turns(tmp_path):
    result, _, rows = run(tmp_path, debate(), trigger="stall")
    assert result["report"]["trigger_observed"] == "three_conflicting_critiques"
    assert len([e for e in rows if e["type"] == "contribution" and e["phase"] == "ordinary_discussion"]) == 4


def test_criticism_of_an_old_proposal_does_not_trigger_current_recovery(tmp_path):
    turns = 0
    def stale_criticism(req):
        nonlocal turns
        if req.agent_id == "_preflight": return '{"ok":true}'
        turns += 1
        ids = re.findall(r"\[id=([^\]]+)\]", req.messages[-1].content)
        active = re.search(r"The active proposal is id=([^,]+), claim=", req.messages[0].content)
        return decision("REVISE" if turns <= 2 else ("REFUTE" if turns == 3 else "RATIFY"),
            "10/19", ids[0] if turns == 3 else (active[1] if active else None))
    result, broker, _ = run(tmp_path, stale_criticism)
    assert result["exit_code"] == 0 and not result["report"]["triggered"]
    assert not any(c.agent_id == RESOLVER for c in broker.backend.calls)


@pytest.mark.parametrize("where", ["sandbox", "live"])
def test_no_false_recovery_when_peers_do_not_endorse(tmp_path, where):
    result, _, _ = run(tmp_path, debate(deny_sandbox=where == "sandbox", deny_live=where == "live"))
    assert result["exit_code"] == 2 and not result["report"]["recovery_confirmed"]
    assert result["report"]["promoted"] == (where == "live")
    assert result["report"]["sandbox_approved"] == (where == "live")


def test_unanimous_wrong_recovery_stays_incorrect(tmp_path):
    result, _, _ = run(tmp_path, debate(wrong_candidate=True))
    assert result["report"]["recovery_confirmed"] and result["report"]["task_correct"] is False
    assert not result["report"]["causal_improvement_claim"]


@pytest.mark.parametrize("resolver_claim", ["2/3", "10/19"])
def test_unanimous_peer_revision_is_promoted_with_its_real_author(tmp_path, resolver_claim):
    base = debate(wrong_candidate=resolver_claim == "2/3")
    sandbox_turns = 0
    def corrected(req):
        nonlocal sandbox_turns
        if req.agent_id != RESOLVER and current_scope().get("phase") == "candidate_replay":
            sandbox_turns += 1
            active = re.search(r"The active proposal is id=([^,]+), claim=", req.messages[0].content)
            return decision("REVISE" if sandbox_turns == 1 else "RATIFY", "10/19", active[1])
        if sandbox_turns and current_scope().get("phase") == "deliberation":
            active = re.search(r"The active proposal is id=([^,]+), claim=", req.messages[0].content)
            return decision("RATIFY", "10/19", active[1])
        return base(req)
    result, _, rows = run(tmp_path, corrected)
    report = result["report"]
    assert report["sandbox_approved"] and report["recovery_confirmed"]
    assert not report["resolver_candidate_approved"]
    assert report["sandbox_final_answer"] == "10/19" and report["sandbox_isolated"]
    posts = [e["contribution"] for e in rows if e["type"] == "contribution"]
    promotion = next(e["contribution"] for e in rows if e.get("phase") == "live_promotion")
    approved = next(c for c in posts if c["contribution_id"] == report["sandbox_approved_proposal_id"])
    assert promotion["agent_id"] == approved["agent_id"] == "general_reasoner"
    assert not promotion["is_counterfactual"]
    assert promotion["metadata"]["peer_revised"]
    assert promotion["metadata"]["sandbox_source_id"] == approved["contribution_id"]
    assert promotion["metadata"]["resolver_proposal_id"] == report["sandbox_proposal_id"]
    saved = prototype_artifacts.read_jsonl(tmp_path / "run" / "results.jsonl")[0]
    assert saved["final_answer"] == "10/19" and saved["correct"]


def test_peer_correction_without_all_votes_stays_in_sandbox(tmp_path):
    base = debate(wrong_candidate=True)
    turns = 0
    def incomplete(req):
        nonlocal turns
        if req.agent_id != RESOLVER and current_scope().get("phase") == "candidate_replay":
            turns += 1
            active = re.search(r"The active proposal is id=([^,]+), claim=", req.messages[0].content)
            return decision("REVISE" if turns == 1 else "RATIFY", "10/19", active[1])
        return base(req)
    result, _, _ = run(tmp_path, incomplete, max_turns=3)
    assert result["report"]["sandbox_final_answer"] == "10/19"
    assert not result["report"]["sandbox_approved"] and not result["report"]["promoted"]
    saved = prototype_artifacts.read_jsonl(tmp_path / "run" / "results.jsonl")[0]
    assert saved["final_answer"] == "2/3" and not saved["correct"]


def test_independent_checks_hide_history_and_counter_a_copied_criticism(tmp_path):
    base = debate()
    critic_failed_once = False
    def anchoring(req):
        nonlocal critic_failed_once
        phase = current_scope().get("phase")
        if phase == "independent_verification":
            text = req.messages[-1].content
            assert "Biased-host probability question" in text
            assert "0 posts" in text and "CURRENT LEADING CLAIM" not in text
            assert "MISTAKEN DIVISION" not in text and "1/19" not in text
            assert "10/19" not in text and "2/3" not in text
            return decision("REVISE", "10/19")
        if phase == "candidate_replay" and req.agent_id != RESOLVER:
            active = re.search(r"The active proposal is id=([^,]+), claim=", req.messages[0].content)
            assert '"claim": "10/19"' in req.messages[0].content
            if req.agent_id == "general_critic" and not critic_failed_once:
                critic_failed_once = True
                return decision("REJECT", "1/19", active[1]).replace(
                    "Likelihood calculation and an explicit observation check.", "MISTAKEN DIVISION gives 1/19.")
            return decision("RATIFY", "10/19", active[1])
        return base(req)
    result, broker, rows = run(tmp_path, anchoring)
    assert result["report"]["recovery_confirmed"] and result["report"]["task_correct"]
    checks = [e for e in rows if e["type"] == "independent_analysis"]
    assert len(checks) == 6 and {e["agent_id"] for e in checks} == {
        "general_reasoner", "general_critic", "prototype_verifier"}
    assert all(e["prediction"] == "10/19" for e in checks)
    assert sum(c.phase == "independent_verification" for c in broker.ledger.records()) == 6
    assert result["manifest"]["prompt_policy"] == "prototype-recovery-v6-calculation-chains"


def test_reference_changes_do_not_change_inference_inputs(tmp_path):
    a, ba, _ = run(tmp_path / "a", debate(), refs={"host": "SECRET_A"})
    b, bb, _ = run(tmp_path / "b", debate(), refs={"host": "SECRET_B"})
    def prompts(backend):
        return [re.sub(r"[0-9a-f]{8}-[0-9a-f-]{27}|[0-9a-f]{32}", "UUID", "\n".join(m.content for m in c.messages)) for c in backend.calls]
    assert prompts(ba.backend) == prompts(bb.backend)
    assert "SECRET_A" not in (tmp_path / "a" / "run" / "trace.jsonl").read_text()


def test_live_stale_endorsement_and_false_division_are_repaired(tmp_path):
    base = debate()
    entered, live_attempts = False, 0
    def reproduce(req):
        nonlocal entered, live_attempts
        if req.agent_id == "_preflight": return '{"ok":true}'
        phase = current_scope()["phase"]
        active = re.search(r"The active proposal is id=([^,]+), claim=", req.messages[0].content)
        if not entered and req.agent_id == "general_critic":
            return decision("REFUTE", "1/19", active[1])
        if req.agent_id == RESOLVER:
            entered = True
        elif entered and phase == "deliberation" and req.agent_id == "general_reasoner":
            live_attempts += 1
            if live_attempts == 1:
                old = re.search(r"\[id=([^\]]+)\] turn 1 \| general_critic \| REFUTE", req.messages[-1].content)
                return decision("RATIFY", "1/19", old[1])
            if live_attempts == 2:
                assert "superseded proposal or criticism" in req.messages[-1].content
                bad = json.loads(decision("REVISE", "1/19", active[1]))
                bad["explanation"]["rationale"] = "(1/3) / (19/30) = 1/19."
                bad["explanation"]["evidence"] = ["CALC: (1/3) / (19/30) = 1/19"]
                return json.dumps(bad)
            assert "= 10/19, not 1/19" in req.messages[-1].content
        return base(req)
    result, _, rows = run(tmp_path, reproduce)
    assert result["report"]["recovery_confirmed"] and result["report"]["task_correct"]
    live = [e["contribution"] for e in rows if e["type"] == "contribution" and e["phase"] == "live_confirmation"]
    assert len(live) == 3 and all(c["payload"]["prediction"]["claim"] == "10/19" for c in live)
    assert [r.split(":")[0] for r in live[0]["metadata"]["repair_reasons"]] == ["stale_ratification", "arithmetic_mismatch"]
    assert len(live[0]["metadata"]["raw_responses"]) == 3
    saved = prototype_artifacts.read_jsonl(tmp_path / "run" / "results.jsonl")[0]
    assert saved["usage"]["repairs"] == 2 and saved["usage"]["backend_attempts"] == 17


@pytest.mark.parametrize("max_calls,expected_outcome", [(100, "inference_failure"), (11, "call_limit")])
def test_bad_live_arithmetic_cannot_overwrite_promoted_answer(tmp_path, max_calls, expected_outcome):
    base = debate()
    entered = False
    def persist(req):
        nonlocal entered
        if req.agent_id == RESOLVER: entered = True
        if entered and current_scope().get("phase") == "deliberation":
            active = re.search(r"The active proposal is id=([^,]+), claim=", req.messages[0].content)
            bad = json.loads(decision("REVISE", "1/19", active[1]))
            bad["explanation"]["evidence"] = ["CALC: (1/3) / (19/30) = 1/19"]
            return json.dumps(bad)
        return base(req)
    result, _, rows = run(tmp_path, persist, max_calls=max_calls)
    saved = prototype_artifacts.read_jsonl(tmp_path / "run" / "results.jsonl")[0]
    assert saved["outcome"] == expected_outcome and saved["final_answer"] == "10/19"
    assert saved["provisional"] and not result["report"]["recovery_confirmed"]
    assert not any(e["type"] == "contribution" and e["phase"] == "live_confirmation" for e in rows)


def test_shared_budget_and_cancel_preserve_partial_natural_history(tmp_path):
    result, _, rows = run(tmp_path / "budget", debate(), max_calls=2)
    assert result["exit_code"] == 2 and not result["report"]["promoted"]
    saved = prototype_artifacts.read_jsonl(tmp_path / "budget" / "run" / "results.jsonl")[0]
    assert saved["outcome"] == "call_limit" and saved["usage"]["backend_attempts"] == 2
    result, _, rows = run(tmp_path / "cancel", debate(interrupt=True))
    assert result["exit_code"] == 130 and not result["report"]["promoted"]
    assert len([e for e in rows if e["type"] == "contribution"]) == 2
    assert (tmp_path / "cancel" / "run" / "demo.html").exists()


def test_challenging_references_follow_the_stated_sampling_rules():
    root = Path(__file__).resolve().parents[1]
    tasks, refs = load_inputs(root / "examples/prototype/challenging_tasks.json", root / "examples/prototype/challenging_references.json")
    assert len(tasks) == 5
    assert F(1, 3) / (F(1, 3) * F(9, 10) + F(1, 3)) == F(refs["biased_host"])
    observed = both = 0
    for genders in itertools.product(["B", "G"], repeat=2):
        for birthdays in itertools.product(range(7), repeat=2):
            for selected in range(2):
                if genders[selected] == "B" and birthdays[selected] == 0:
                    observed += 1; both += genders == ("B", "B")
    assert F(both, observed) == F(refs["sampled_child"])
    assert (F(9, 10) * F(1, 10)) / (F(9, 10) * F(1, 10) + F(1, 10) * F(9, 10)) == F(refs["correlated_detectors"])
    alternatives = [other for s in (10, 20) for own, other in ((s, 2*s), (2*s, s)) if own == 20]
    assert sum(alternatives) / len(alternatives) == refs["two_envelopes"]
    assert F(9, 10) > F(80, 100) and F(40, 100) > F(3, 10) and F(49, 110) < F(83, 110)
    assert refs["load_mix"] == "A"
