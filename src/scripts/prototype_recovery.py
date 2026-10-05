"""Fresh live agents, observed disagreement, isolated counterfactual, live confirmation.

No starting history is seeded. Ollama generates every contribution. Mock mode is
an explicitly scripted test. Disagreement and recovery are measured, not promised.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from uuid import uuid4

from agents.base import AgentTurnError
from agents.board_client import InMemoryBoard
from contracts.schemas import AgentRole, PXPTag
from llm_broker import InferenceSettings, build_broker, usage_scope
from llm_broker.backends.mock import MockBackend
from llm_broker.budget import BudgetExceeded, CallBudget, budget_scope
from llm_broker.preflight import run_preflight
from prompts.personas import Persona
from prompts.validation import normalize_claim
from scripts.prototype_artifacts import export_artifacts
from scripts.prototype_validation import VALIDATION_POLICY, PrototypeAgent
from scripts.prototype_run import AgreementTracker, PROMPT_POLICY, PrototypeTask, RunWriter, TrialOutcome, grade, load_inputs, make_panel, revision, turn_instructions

RESOLVER = "counterfactual_resolver"
RECOVERY_PROMPT_POLICY = "prototype-recovery-v7-fraction-grouping"


def resolver_agent(broker, settings, seed):
    persona = Persona(id=RESOLVER, domain="general", role=AgentRole.COUNTERFACTUAL,
        title="Counterfactual Resolver", expertise="Testing alternative assumptions in an isolated historical replay.",
        stance="Investigate the unsupported assumption behind disagreement. Derive an alternative from the original task; peers must check it independently.")
    return PrototypeAgent(RESOLVER, persona, broker, model=settings.model, seed=seed + 3,
        counterfactual_capable=True, reasoning_first=True, **settings.agent_kwargs())


def mock_recovery_backend():
    entered = False
    def respond(request):
        nonlocal entered
        if request.agent_id == "_preflight": return '{"ok":true}'
        active = re.search(r"The active proposal is id=([^,]+), claim=", request.messages[0].content)
        if request.agent_id == RESOLVER: entered = True
        tag = "REVISE" if not active or request.agent_id == RESOLVER else ("RATIFY" if entered else "REFUTE")
        claim = "SCRIPTED RECOVERY ANSWER" if entered else "SCRIPTED INITIAL ANSWER"
        return json.dumps({"explanation": {"rationale": "Scripted protocol exercise, not model reasoning or benchmark performance.", "evidence": []},
            "prediction": {"claim": claim, "confidence": .5}, "tag": tag,
            "target_contribution_id": active[1] if active else None})
    return MockBackend(responder=respond)


def frozen_hash(board, sid):
    return hashlib.sha256(board.get_snapshot(sid).model_dump_json().encode()).hexdigest()


def run_recovery(task: PrototypeTask, out: Path, *, settings: InferenceSettings, seed: int = 42,
                 initial_turns: int = 9, max_turns: int = 6, max_calls: int = 100, timeout_s: float = 1200,
                 trigger: str = "first-conflict", require_gpu: bool = False,
                 references: dict | None = None, broker=None):
    if min(initial_turns, max_turns, max_calls, timeout_s) <= 0: raise ValueError("limits must be positive")
    if trigger not in {"first-conflict", "stall"}: raise ValueError("invalid recovery trigger")
    created = broker is None
    broker = broker or build_broker(settings)
    if created and settings.backend == "mock": broker.backend = mock_recovery_backend()
    if broker.backend.name != settings.backend: raise ValueError("configured backend mismatch")
    mode = "scripted" if settings.backend == "mock" else "live"
    writer = RunWriter(Path(out))
    panel = make_panel(broker, settings, seed)
    cf_agent = resolver_agent(broker, settings, seed)
    voters = [agent.agent_id for agent in panel]
    main = InMemoryBoard()
    main_sid = main.create_session(task.task_id, task.question, copy.deepcopy(task.context)).session_id
    tracker = AgreementTracker(voters, proposers=[RESOLVER])
    trial_id = uuid4().hex + ":recovery-demo"
    manifest = {"manifest_version": 1, "kind": "natural_counterfactual_demo", "mode": mode,
        "status": "preflight", "source": revision(), "settings": settings.to_dict(),
        "prompt_policy": RECOVERY_PROMPT_POLICY, "base_prompt_policy": PROMPT_POLICY,
        "arithmetic_validation": VALIDATION_POLICY,
        "tasks": [task.model_dump()], "seeds": [seed],
        "recovery_enabled": True, "recovery_trigger": trigger, "sandbox_promotion_policy": "latest_unanimous_proposal",
        "history_origin": "live model" if mode == "live" else "scripted test",
        "agents": [agent.spec() for agent in panel] + [cf_agent.spec()],
        "limits": {"ordinary_turns": initial_turns, "peer_turns_per_phase": max_turns,
                   "max_backend_attempts_total": max_calls, "timeout_s_total": timeout_s},
        "started_at": datetime.now(timezone.utc).isoformat()}
    writer.json("manifest.json", manifest)
    active = {"phase": "preflight"}
    previous_hook = broker.response_hook
    broker.ledger.attach_file(writer.root / "calls.jsonl")
    def raw(request, response):
        writer.event("raw_response", trial_id=trial_id, phase=active["phase"], agent_id=request.agent_id,
            is_repair=bool(request.metadata.get("is_repair")), text=response.text,
            messages=[m.model_dump() for m in request.messages])
        if previous_hook: previous_hook(request, response)
    broker.response_hook = raw
    report = {"trigger_policy": trigger, "trigger_observed": None, "triggered": False, "rollback_before_turn": None,
              "rollback_policy": "before_disputed_active_proposal",
              "sandbox_approved": False, "promoted": False, "recovery_confirmed": False,
              "sandbox_agreement": False, "sandbox_final_answer": None,
              "resolver_candidate_approved": False, "sandbox_approved_proposal_id": None,
              "sandbox_approved_agent_id": None,
              "independent_checks": 0,
              "causal_improvement_claim": False, "paired_baseline_evaluated": False}
    stats = {"attempted": 0, "succeeded": 0}
    outcome, error, exit_code = "not_started", None, 0
    budget = None
    started = False

    def post_event(contribution, board, sid, votes, phase, origin):
        writer.event("contribution", trial_id=trial_id, phase=phase, origin=origin,
            contribution=contribution.model_dump(mode="json"), agreeing_agents=sorted(votes.votes),
            board_version=board.get_snapshot(sid).version)

    def act(agent, board, sid, votes, phase, extra=""):
        active["phase"] = phase
        stats["attempted"] += 1
        budget.check()
        snapshot = board.get_snapshot(sid)
        if votes.active: snapshot.active_claim = votes.active.payload.prediction.model_copy(deep=True)
        writer.event("turn_started", trial_id=trial_id, phase=phase, agent_id=agent.agent_id, board_version=snapshot.version)
        contribution = agent.act(snapshot, extra_instructions=turn_instructions(votes.active) + " " + extra,
                                 must_keep_ids=[votes.active.contribution_id] if votes.active else [])
        if budget.remaining() <= 0: raise BudgetExceeded("time_limit")
        if agent.agent_id == RESOLVER:
            if contribution.tag != PXPTag.REVISE:
                raise AgentTurnError("A counterfactual replacement must be REVISE", agent_id=RESOLVER)
            contribution.is_counterfactual = True
        board.submit(contribution); votes.add(contribution)
        stats["succeeded"] += 1
        post_event(contribution, board, sid, votes, phase, "model" if mode == "live" else "scripted")
        return contribution

    def independent_check(agent, board, sid, phase):
        # Hide *all* candidate claims and dialogue. Keep only the original task
        # and context, so a critic's arithmetic error cannot anchor this call.
        blind = board.get_snapshot(sid).model_copy(deep=True, update={
            "contributions": [], "active_claim": None, "active_agents": [], "version": 0})
        active["phase"] = phase + "_independent_check"
        stats["attempted"] += 1
        budget.check()
        methods = {
            "general_reasoner": "Derive the answer directly from the stated constraints.",
            "general_critic": "Use a likelihood table: for each possible underlying state, compute prior times observation likelihood, then normalize the requested state's weight by the total weight.",
            "prototype_verifier": "Use a separate check by enumerating weighted possible worlds or computing posterior odds. Verify that posterior probabilities sum to one.",
        }
        with usage_scope(phase="independent_verification"):
            checked = agent.act(blind, extra_instructions=turn_instructions(None) +
                " INDEPENDENT CHECK: this is private scratch work before seeing any debate. " + methods[agent.agent_id] +
                " For a conditional probability, show the numerator weight and denominator weight explicitly. "
                "Check the division with exact fractions. Do not subtract an unobserved state's weight or drop a term. "
                "Put your own conclusion in prediction.claim. This scratch work is not a board submission.")
        if budget.remaining() <= 0: raise BudgetExceeded("time_limit")
        stats["succeeded"] += 1
        report["independent_checks"] += 1
        writer.event("independent_analysis", trial_id=trial_id, phase=phase + "_independent_check",
            agent_id=agent.agent_id, prediction=checked.prediction,
            rationale=checked.payload.explanation.rationale, origin="model" if mode == "live" else "scripted")
        return checked

    def peers(board, sid, votes, phase):
        checks = {}
        for turn in range(max_turns):
            agent = panel[turn % len(panel)]
            if agent.agent_id not in checks:
                checks[agent.agent_id] = independent_check(agent, board, sid, phase)
            checked = checks[agent.agent_id]
            act(agent, board, sid, votes, phase,
                "YOUR PRIOR INDEPENDENT CHECK (model-generated from the original question, not an oracle): " +
                json.dumps({"claim": checked.prediction, "rationale": checked.payload.explanation.rationale,
                            "calculator_checks": checked.metadata.get("arithmetic_checks", [])}) +
                " Compare the active proposal with this calculation. If they differ, recalculate the exact arithmetic "
                "and explain which step is wrong. Neither your scratch answer nor the board is authoritative. "
                "Do not copy the latest critic's number merely because it is recent. "
                "A REFUTE/REJECT post is criticism, not a replacement proposal. If you have a better computed "
                "answer, submit it with REVISE; if the active proposal survives your checks, RATIFY it.")
            if votes.agreed or votes.stalled: break

    def recover(rollback):
        report["triggered"], report["rollback_before_turn"] = True, rollback
        writer.event("recovery_stage", trial_id=trial_id, phase="disagreement_detected",
            message="Observed agent disagreement; counterfactual resolver enters under the configured trigger policy.")
        saved = main.get_snapshot(main_sid)
        report["main_hash_before_sandbox"] = frozen_hash(main, main_sid)
        sandbox = InMemoryBoard()
        sandbox_sid = sandbox.create_session(task.task_id, task.question, copy.deepcopy(task.context)).session_id
        sandbox_tracker = AgreementTracker(voters, proposers=[RESOLVER])
        for old in saved.contributions[:rollback]:
            cloned = old.model_copy(deep=True, update={"session_id": sandbox_sid})
            sandbox.submit(cloned); sandbox_tracker.add(cloned)
            post_event(cloned, sandbox, sandbox_sid, sandbox_tracker, "sandbox_checkpoint", "replayed_model")
        writer.event("recovery_stage", trial_id=trial_id, phase="sandbox",
            message=f"Isolated historical replay: rewind before turn {rollback} and generate an alternative proposal.")
        # Diagnostics use only already-observed agent text, never evaluator references.
        diagnostics = "\n".join(f"{c.agent_id} {c.tag.value}: {c.prediction}; {c.payload.explanation.rationale[:400]}"
                                for c in saved.contributions[-4:])
        with usage_scope(phase="candidate_replay"):
            candidate = act(cf_agent, sandbox, sandbox_sid, sandbox_tracker, "sandbox_alternative",
                "COUNTERFACTUAL STEP: replace the earlier commitment with a new REVISE. "
                "Ask what would change if unsupported assumptions were checked before that commitment. "
                "Use the original task and the observed diagnostics below; do not force agreement. "
                "Do not use ids from the removed future history as targets. Observed disagreement:\n" + diagnostics)
            report["sandbox_proposal_id"] = candidate.contribution_id
            peers(sandbox, sandbox_sid, sandbox_tracker, "sandbox_peer_check")
        # Peers may correct the resolver's answer. Approve the exact final root
        # endorsed by all three, preserving its actual author and provenance.
        approved = sandbox_tracker.active
        report["sandbox_agreement"] = sandbox_tracker.agreed
        report["sandbox_final_answer"] = approved.prediction if approved else None
        report["sandbox_approved"] = sandbox_tracker.agreed
        report["resolver_candidate_approved"] = sandbox_tracker.agreed and approved.contribution_id == candidate.contribution_id
        if report["sandbox_approved"]:
            report["sandbox_approved_proposal_id"] = approved.contribution_id
            report["sandbox_approved_agent_id"] = approved.agent_id
        report["main_hash_after_sandbox"] = frozen_hash(main, main_sid)
        report["sandbox_isolated"] = report["main_hash_before_sandbox"] == report["main_hash_after_sandbox"]
        if not report["sandbox_isolated"]: raise RuntimeError("sandbox changed live state")
        if not report["sandbox_approved"]: return "sandbox_not_approved"
        promoted = approved.model_copy(deep=True, update={"contribution_id": uuid4().hex,
            "session_id": main_sid, "turn_index": len(saved.contributions),
            "target_contribution_id": tracker.active.contribution_id,
            "metadata": {"origin": "sandbox_promotion", "sandbox_source_id": approved.contribution_id,
                         "resolver_proposal_id": candidate.contribution_id,
                         "peer_revised": approved.contribution_id != candidate.contribution_id}})
        main.submit(promoted); tracker.add(promoted)
        report["promoted"], report["promoted_proposal_id"] = True, promoted.contribution_id
        writer.event("recovery_stage", trial_id=trial_id, phase="live_confirmation",
            message="All three peers endorsed the sandbox proposal. Promote it and check again on the live board.")
        post_event(promoted, main, main_sid, tracker, "live_promotion", "promoted_model" if mode == "live" else "scripted")
        with usage_scope(phase="deliberation"):
            peers(main, main_sid, tracker, "live_confirmation")
        report["recovery_confirmed"] = tracker.agreed and tracker.active.contribution_id == promoted.contribution_id
        return "recovery_confirmed" if report["recovery_confirmed"] else "live_recovery_not_confirmed"

    try:
        preflight = run_preflight(broker, settings, probe=True)
        manifest["preflight"], manifest["model_digest"] = preflight, preflight.get("model_digest")
        gpu_ok = bool(preflight["hardware"]["gpus"]) and any(c["name"] == "model_fully_on_gpu" and c["ok"] for c in preflight["checks"])
        if not preflight["ok"] or (require_gpu and not gpu_ok):
            manifest["status"], exit_code = "preflight_failed", 1
        else:
            manifest["status"] = "running"; writer.json("manifest.json", manifest)
            budget, started = CallBudget(max_calls, timeout_s), True
            writer.event("trial_started", trial_id=trial_id, task=task.model_dump(), seed=seed,
                agents=[a.spec() for a in panel], session_id=main_sid)
            with usage_scope(trial_id=trial_id, phase="deliberation"), budget_scope(budget):
                rollback = None
                for turn in range(initial_turns):
                    previous = tracker.active
                    c = act(panel[turn % len(panel)], main, main_sid, tracker, "ordinary_discussion")
                    criticizes_active = previous is not None and c.tag in {PXPTag.REFUTE, PXPTag.REJECT} and tracker.roots.get(c.target_contribution_id or "") == previous.contribution_id
                    revises_answer = previous is not None and c.tag == PXPTag.REVISE and normalize_claim(c.prediction) != normalize_claim(previous.prediction)
                    conflict = previous is not None and c.agent_id != previous.agent_id and (criticizes_active or revises_answer)
                    if tracker.agreed:
                        outcome = "ordinary_agreement"; break
                    if (trigger == "first-conflict" and conflict) or tracker.stalled:
                        report["trigger_observed"] = "first_agent_disagreement" if trigger == "first-conflict" else "three_conflicting_critiques"
                        rollback = previous.turn_index if trigger == "first-conflict" else tracker.active.turn_index
                        break
                else:
                    outcome = "ordinary_turn_limit"
                if rollback is not None: outcome = recover(rollback)
            exit_code = 0 if outcome in {"ordinary_agreement", "recovery_confirmed"} else 2
            manifest["status"] = "completed" if exit_code == 0 else "completed_without_agreement"
    except KeyboardInterrupt:
        outcome, error, exit_code, manifest["status"] = "cancelled", "Interrupted by user", 130, "cancelled"
    except (AgentTurnError, BudgetExceeded) as e:
        limit = e if isinstance(e, BudgetExceeded) else e.__cause__
        outcome = limit.reason if isinstance(limit, BudgetExceeded) else "inference_failure"
        error, exit_code, manifest["status"] = str(e), 2, "failed"
        payload = e.to_record() if isinstance(e, AgentTurnError) else {"error": error}
        writer.event("turn_failed", trial_id=trial_id, phase=active["phase"], **payload)
    except Exception as e:
        outcome, error, exit_code, manifest["status"] = "internal_failure", f"{type(e).__name__}: {e}", 1, "failed"
    finally:
        if started:
            usage = broker.trial_usage(trial_id)
            answer = tracker.active.prediction if tracker.active else None
            grading, correct = grade(answer, task.task_id, references or {})
            result = TrialOutcome(trial_id=trial_id, task_id=task.task_id, seed=seed, mode=mode,
                outcome=outcome, final_answer=answer, provisional=not tracker.agreed,
                answer_agreement=tracker.agreed, agreeing_agents=sorted(tracker.votes),
                grading=grading, correct=correct, turns_attempted=stats["attempted"], turns_succeeded=stats["succeeded"],
                turns_failed=stats["attempted"] - stats["succeeded"], usage={**usage.model_dump(), "total_tokens": usage.total_tokens},
                duration_seconds=round(time.monotonic() - budget.started, 3), recovery_enabled=True, error=error)
            writer.append("results.jsonl", result.model_dump())
            writer.event("trial_finished", trial_id=trial_id, result=result.model_dump())
            report["task_correct"] = correct
        if error: manifest["error"] = error
        manifest["recovery"] = report
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        writer.json("manifest.json", manifest); writer.json("recovery.json", report)
        broker.response_hook = previous_hook; broker.ledger.attach_file(None); broker.close()
        try: export_artifacts(writer.root)
        except Exception as e:
            manifest["export_error"] = str(e); writer.json("manifest.json", manifest)
            if not exit_code: exit_code = 1
    return {"exit_code": exit_code, "manifest": manifest, "report": report, "out": str(out)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["ollama", "mock"], default="ollama")
    parser.add_argument("--model")
    parser.add_argument("--tasks", type=Path, default=Path("examples/prototype/challenging_tasks.json"))
    parser.add_argument("--references", type=Path)
    parser.add_argument("--task-id", default=None, help="select one task; default is the first question")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-turns", type=int, default=9)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--max-calls", type=int, default=100)
    parser.add_argument("--timeout-s", type=float, default=1200)
    parser.add_argument("--trigger", choices=["first-conflict", "stall"], default="first-conflict")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        tasks, refs = load_inputs(args.tasks, args.references)
        task = tasks[0] if args.task_id is None else next((task for task in tasks if task.task_id == args.task_id), None)
        if task is None: raise ValueError("unknown task id")
        result = run_recovery(task, args.out,
            settings=InferenceSettings.from_env(backend=args.backend, model=args.model, max_concurrency=1),
            seed=args.seed, initial_turns=args.initial_turns, max_turns=args.max_turns, max_calls=args.max_calls,
            timeout_s=args.timeout_s, trigger=args.trigger, require_gpu=args.require_gpu, references=refs)
    except (OSError, ValueError) as e:
        print(f"Cannot start recovery run: {e}", file=sys.stderr); return 1
    print(f"Artifacts: {args.out.resolve()}\nDisagreement observed: {result['report']['triggered']}\nRecovery confirmed: {result['report']['recovery_confirmed']}")
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
