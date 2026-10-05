"""Text-only presentation prototype: real questions, measured trials, no recovery.

python -m scripts.prototype_run --backend mock --tasks examples/prototype/tasks.json \
    --references examples/prototype/references.json --out outputs/presentation/offline
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agents.base import AgentTurnError
from agents.board_client import InMemoryBoard
from contracts.schemas import AgentContribution, AgentRole, PXPTag
from ingest.base import SOLUTION_KEYS
from llm_broker import InferenceSettings, ModelBroker, build_broker, usage_scope
from llm_broker.backends.mock import MockBackend
from llm_broker.budget import CallBudget, budget_scope
from llm_broker.preflight import run_preflight
from prompts.personas import Persona, get_persona
from prompts.validation import normalize_claim
from scripts.prototype_artifacts import export_artifacts
from scripts.prototype_validation import CALC_INSTRUCTIONS, VALIDATION_POLICY, PrototypeAgent


PROMPT_POLICY = "prototype-v6-calculation-chains"


def turn_instructions(active: AgentContribution | None) -> str:
    """Prototype-specific verification and endorsement rules; no reference access."""
    rules = (
        "Keep prediction.claim to the requested final-answer format. "
        "Independently solve the original question before choosing a tag. "
        "For numerical questions, define the unknowns, express ALL given constraints as equations, "
        "solve them, and substitute your proposed values back into EVERY original constraint. "
        "Put the key calculation and substitution check at the START of explanation.rationale "
        "so other agents can inspect them. For logical questions, check whether a counterexample exists. "
        "Choose prediction.claim from the conclusion of your OWN calculation. "
        "Before submitting, check that prediction.claim agrees with explanation.rationale. "
        "If your reasoning establishes a corrected answer, post that answer with REVISE targeting "
        "the active proposal. Do not keep the disputed target's claim while explaining a different answer. "
        "Use REFUTE or REJECT only when you can identify a problem but cannot yet supply a corrected answer. "
        "Check the current proposal against those constraints, not against a vote count. "
        "PROTOTYPE ENDORSEMENT RULE (overrides the general rule about revising when persuaded): "
        "If you accept the active claim AND its explanation, use RATIFY, even when this changes "
        "your earlier view or you authored the proposal. Restate its claim exactly and target its id. "
        "A different wording of sound reasoning is not a reason to REVISE. "
        "Use REVISE only to correct the answer or a substantive reasoning error, and identify "
        "the exact correction. Every REVISE starts a new proposal and clears all endorsement votes. "
        "Do not repeat a criticism without checking the original question and giving concrete evidence. "
        "Never endorse an answer just to finish the discussion. "
        "OUTPUT ORDER (overrides the earlier example's key order): write explanation FIRST, "
        "then prediction based on that explanation, then tag and target_contribution_id. "
    )
    rules += CALC_INSTRUCTIONS
    rules += "Return only this JSON structure, choosing the appropriate tag (populate evidence with CALC entries for numerical answers): " + json.dumps({
        "explanation": {"rationale": "calculation and conclusion", "evidence": []},
        "prediction": {"claim": "your computed final answer", "confidence": .8},
        "tag": "REVISE|RATIFY|REFUTE|REJECT",
        "target_contribution_id": active.contribution_id if active else None,
    }) + ". "
    if active is None:
        return rules + "The board is empty: post your independently checked hypothesis with REVISE and target null."
    return rules + (
        f"The active proposal is id={active.contribution_id}, "
        f"claim={json.dumps(active.prediction)}. "
        "If its claim AND explanation survive your checks, explicitly RATIFY that id with its claim. "
        "Otherwise correct it with REVISE or criticize with concrete reasons."
    )


class PrototypeTask(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    task_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    context: dict[str, Any] = Field(default_factory=dict)

    @field_validator("task_id", "question")
    @classmethod
    def nonblank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("task identifiers and questions cannot be blank")
        return v.strip()

    @field_validator("context")
    @classmethod
    def no_solutions(cls, v: dict) -> dict:
        def check(item):
            if isinstance(item, dict):
                for key, value in item.items():
                    if key.lower() in SOLUTION_KEYS:
                        raise ValueError(f"evaluator-only field {key!r} is forbidden in context")
                    check(value)
            elif isinstance(item, list):
                for value in item:
                    check(value)
        check(v)
        return v


class TrialOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")
    result_version: int = 1
    trial_id: str
    task_id: str
    seed: int
    mode: str
    outcome: str
    final_answer: str | None
    provisional: bool
    answer_agreement: bool
    agreeing_agents: list[str]
    grading: str
    correct: bool | None
    turns_attempted: int
    turns_succeeded: int
    turns_failed: int
    usage: dict
    duration_seconds: float
    recovery_enabled: bool = False
    error: str | None = None


def load_inputs(tasks_file: Path, references_file: Path | None):
    raw = json.loads(tasks_file.read_text())
    if not isinstance(raw, list) or not raw:
        raise ValueError("tasks must be a non-empty JSON list")
    tasks = [PrototypeTask.model_validate(row) for row in raw]
    if len({t.task_id for t in tasks}) != len(tasks):
        raise ValueError("duplicate task_id")
    refs = json.loads(references_file.read_text()) if references_file else {}
    if not isinstance(refs, dict):
        raise ValueError("references must be an object keyed by task_id")
    for key, value in refs.items():
        if key not in {t.task_id for t in tasks}:
            raise ValueError(f"reference for unknown task {key!r}")
        if isinstance(value, dict):
            if set(value) != {"manual_review"} or value["manual_review"] is not True:
                raise ValueError("manual references must be {\"manual_review\": true}")
        else:
            answers = value if isinstance(value, list) else [value]
            if not answers or any(not isinstance(a, (str, int, float, bool)) or
                                  not str(a).strip() or (isinstance(a, float) and not math.isfinite(a)) for a in answers):
                raise ValueError(f"invalid reference for {key!r}")
    return tasks, refs


def normalize_answer(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().casefold().rstrip(".!;,").strip()


def grade(answer: str | None, task_id: str, references: dict) -> tuple[str, bool | None]:
    if task_id not in references:
        return "unscored", None
    ref = references[task_id]
    if isinstance(ref, dict):
        return "manual_review", None
    alternatives = ref if isinstance(ref, list) else [ref]
    return "strict_match", bool(answer and normalize_answer(answer)) and any(
        normalize_answer(answer) == normalize_answer(value) for value in alternatives
    )


class AgreementTracker:
    """Latest explicit answer endorsements, bound to one active REVISE root."""
    def __init__(self, agents: list[str], proposers: list[str] | None = None):
        self.agents = set(agents)
        self.proposers = set(proposers or [])
        self.active: AgentContribution | None = None
        self.roots: dict[str, str] = {}
        self.votes: set[str] = set()
        self.conflicts: list[str] = []

    def add(self, contribution: AgentContribution) -> None:
        c = contribution
        if c.agent_id not in self.agents and not (c.agent_id in self.proposers and c.tag == PXPTag.REVISE):
            raise ValueError("contribution outside the fixed roster")
        if c.tag == PXPTag.REVISE:
            self.active = c
            self.roots[c.contribution_id] = c.contribution_id
            self.votes.clear()
            self.conflicts.clear()
            return
        root = self.roots.get(c.target_contribution_id or "")
        self.votes.discard(c.agent_id)
        if c.tag == PXPTag.RATIFY and self.active and root == self.active.contribution_id:
            if normalize_claim(c.prediction) == normalize_claim(self.active.prediction):
                self.votes.add(c.agent_id)
                self.roots[c.contribution_id] = root
                self.conflicts.clear()
                return
        if c.tag in (PXPTag.REFUTE, PXPTag.REJECT) and self.active and root == self.active.contribution_id:
            self.roots[c.contribution_id] = root
            self.conflicts.append(c.agent_id)
        else:
            self.conflicts.clear()

    @property
    def agreed(self) -> bool:
        return self.votes == self.agents

    @property
    def stalled(self) -> bool:
        return len(self.conflicts) >= 3 and len(set(self.conflicts[-3:])) >= 2


class RunWriter:
    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=False)
        self.root = root
        self.sequence = 0
        for name in ("trace.jsonl", "results.jsonl", "calls.jsonl"):
            (root / name).touch(exist_ok=False)

    def append(self, name: str, row: dict) -> None:
        with (self.root / name).open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def json(self, name: str, row: dict) -> None:
        target = self.root / name
        tmp = target.with_suffix(target.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(row, f, indent=2, ensure_ascii=False, allow_nan=False, default=str)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(target)

    def event(self, event_type: str, **payload) -> None:
        self.sequence += 1
        self.append("trace.jsonl", {"event_id": uuid4().hex, "sequence": self.sequence,
            "timestamp": datetime.now(timezone.utc).isoformat(), "type": event_type, **payload})


def make_panel(broker: ModelBroker, settings: InferenceSettings, seed: int):
    personas = [get_persona("general_reasoner"), get_persona("general_critic"), Persona(
        id="prototype_verifier", domain="general", role=AgentRole.DOMAIN_EXPERT,
        title="Independent Verifier", expertise="Independent arithmetic and logical verification.",
        stance="Check the proposed answer independently; endorse only a sound answer and explanation.")]
    return [PrototypeAgent(p.id, p, broker, model=settings.model, seed=seed + i,
                     counterfactual_capable=False, reasoning_first=True,
                     **settings.agent_kwargs()) for i, p in enumerate(personas)]


def scripted_backend() -> MockBackend:
    """Exercise protocol flow without pretending to solve the user's task."""
    def respond(req):
        text = "\n".join(m.content for m in req.messages if m.role == "user")
        ids = re.findall(r"\[id=([^\]]+)\]", text)
        return json.dumps({"tag": "RATIFY" if ids else "REVISE",
            "target_contribution_id": ids[0] if ids else None,
            "prediction": {"claim": "SCRIPTED DEMO ANSWER", "confidence": 0.5},
            "explanation": {"rationale": "Scripted protocol exercise; this is not model reasoning.", "evidence": []}})
    return MockBackend(responder=respond)


def revision() -> dict:
    packaged = Path(__file__).resolve().parents[2] / "prototype_source_manifest.json"
    if packaged.exists():
        data = json.loads(packaged.read_text())
        return {key: data.get(key) for key in ("commit", "working_tree_dirty", "source_digest")}
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True, timeout=5).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                   text=True, check=True, timeout=5).stdout.strip())
        return {"commit": commit, "working_tree_dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "working_tree_dirty": None}


def run_prototype(tasks: list[PrototypeTask], references: dict, out: Path, *,
                  settings: InferenceSettings, seeds: list[int], max_turns: int = 9,
                  max_calls: int = 100, timeout_s: float = 1200, broker: ModelBroker | None = None,
                  require_gpu: bool = False) -> dict:
    if not tasks or len({t.task_id for t in tasks}) != len(tasks):
        raise ValueError("provide unique non-empty tasks")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("provide unique repetition seeds")
    if min(max_turns, max_calls, timeout_s) <= 0:
        raise ValueError("limits must be positive")
    run_id = uuid4().hex
    mode = "scripted" if settings.backend == "mock" else "live"
    created_broker = broker is None
    broker = broker or build_broker(settings)
    if created_broker and mode == "scripted":
        broker.backend = scripted_backend()
    if broker.backend.name != settings.backend:
        raise ValueError("broker/backend does not match the explicitly configured mode")
    writer = RunWriter(Path(out))
    manifest = {"manifest_version": 1, "kind": "text_deliberation_prototype", "run_id": run_id,
        "mode": mode, "status": "preflight", "source": revision(), "settings": settings.to_dict(),
        "prompt_policy": PROMPT_POLICY,
        "arithmetic_validation": VALIDATION_POLICY,
        "tasks": [t.model_dump() for t in tasks], "seeds": seeds, "recovery_enabled": False,
        "limits": {"max_turns": max_turns, "max_backend_attempts": max_calls, "timeout_s": timeout_s},
        "agents": [a.spec() for a in make_panel(broker, settings, seeds[0])],
        "task_hash": hashlib.sha256(json.dumps([t.model_dump() for t in tasks], sort_keys=True).encode()).hexdigest(),
        "started_at": datetime.now(timezone.utc).isoformat()}
    writer.json("manifest.json", manifest)
    previous_hook = broker.response_hook
    active = {"trial_id": None}
    broker.ledger.attach_file(writer.root / "calls.jsonl")
    def raw_response(request, response):
        writer.event("raw_response", trial_id=active["trial_id"], agent_id=request.agent_id,
                     is_repair=bool(request.metadata.get("is_repair")), text=response.text,
                     messages=[m.model_dump() for m in request.messages])
        if previous_hook:
            previous_hook(request, response)
    broker.response_hook = raw_response
    results = []
    exit_code = 0
    try:
        report = run_preflight(broker, settings, probe=True)
        manifest["preflight"] = report
        manifest["model_digest"] = report.get("model_digest")
        gpu_ok = bool(report["hardware"]["gpus"]) and any(
            c["name"] == "model_fully_on_gpu" and c["ok"] for c in report["checks"])
        if not report["ok"] or (require_gpu and not gpu_ok):
            manifest["status"] = "preflight_failed"
            writer.event("preflight_failed", report=report, require_gpu=require_gpu)
            exit_code = 1
        else:
            manifest["status"] = "running"
            writer.json("manifest.json", manifest)
            writer.event("run_started", mode=mode)
            for task in tasks:
                for seed in seeds:
                    active["trial_id"] = f"{run_id}:{task.task_id}:{seed}"
                    result = run_trial(task, seed, broker, settings, writer, active["trial_id"],
                                       max_turns, max_calls, timeout_s, mode)
                    grading, correct = grade(result.final_answer, task.task_id, references)
                    result.grading, result.correct = grading, correct
                    # Only the evaluator, after inference, reads the reference file.
                    writer.append("results.jsonl", result.model_dump())
                    results.append(result.model_dump())
                    writer.event("trial_finished", trial_id=result.trial_id, result=result.model_dump())
                    print(f"{task.task_id} seed={seed}: {result.outcome}; answer={result.final_answer!r}; "
                          f"correct={correct}; tokens={result.usage['total_tokens']}", flush=True)
                    if result.outcome in ("cancelled", "internal_failure"):
                        exit_code = 130 if result.outcome == "cancelled" else 1
                        break
                if exit_code:
                    break
            if not exit_code and any(r["outcome"] == "inference_failure" for r in results):
                exit_code = 2
            manifest["status"] = "cancelled" if exit_code == 130 else ("completed_with_failures" if exit_code else "completed")
    except KeyboardInterrupt:
        manifest["status"] = "cancelled"
        exit_code = 130
        writer.event("run_cancelled")
    except Exception as e:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(e).__name__}: {e}"
        writer.event("run_failed", error=manifest["error"])
        exit_code = 1
    finally:
        broker.response_hook = previous_hook
        broker.ledger.attach_file(None)
        broker.close()
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        writer.json("manifest.json", manifest)
        try:
            export_artifacts(writer.root)
        except Exception as e:
            manifest["export_error"] = f"{type(e).__name__}: {e}"
            writer.json("manifest.json", manifest)
            print(f"Export failed; raw artifacts remain in {writer.root}: {e}", file=sys.stderr)
            if not exit_code:
                exit_code = 1
    return {"exit_code": exit_code, "out": str(writer.root), "manifest": manifest, "results": results}


def run_trial(task, seed, broker, settings, writer, trial_id, max_turns, max_calls, timeout_s, mode):
    board = InMemoryBoard()
    sid = board.create_session(task.task_id, task.question, task.context).session_id
    panel = make_panel(broker, settings, seed)
    tracker = AgreementTracker([a.agent_id for a in panel])
    budget = CallBudget(max_calls, timeout_s)
    attempted = succeeded = failed = 0
    outcome, error = "turn_limit", None
    writer.event("trial_started", trial_id=trial_id, task=task.model_dump(), seed=seed,
                 agents=[a.spec() for a in panel], session_id=sid)
    try:
        with usage_scope(trial_id=trial_id, phase="deliberation", session_id=sid), budget_scope(budget):
            for turn in range(max_turns):
                budget.check()
                agent = panel[turn % 3]
                attempted += 1
                hint = turn_instructions(tracker.active)
                writer.event("turn_started", trial_id=trial_id, agent_id=agent.agent_id, turn=turn,
                             board_version=board.get_snapshot(sid).version)
                try:
                    snapshot = board.get_snapshot(sid)
                    if tracker.active:
                        # The shared stub lets a RATIFY of an older post change
                        # active_claim. Keep this prototype's agent-visible lead
                        # bound to its current REVISE root, as its votes already are.
                        snapshot.active_claim = tracker.active.payload.prediction.model_copy(deep=True)
                    keep = [tracker.active.contribution_id] if tracker.active else []
                    contribution = agent.act(snapshot, extra_instructions=hint, must_keep_ids=keep)
                    if budget.remaining() <= 0:
                        from llm_broker.budget import BudgetExceeded
                        raise BudgetExceeded("time_limit")
                    board.submit(contribution)
                    tracker.add(contribution)
                    succeeded += 1
                    writer.event("contribution", trial_id=trial_id, contribution=contribution.model_dump(mode="json"),
                                 agreeing_agents=sorted(tracker.votes), board_version=board.get_snapshot(sid).version)
                except AgentTurnError as e:
                    failed += 1
                    tracker.votes.discard(agent.agent_id)
                    writer.event("turn_failed", trial_id=trial_id, **e.to_record())
                    # A genuine backend failure terminates this trial. Invalid JSON may retry
                    # on the next ordinary turn, but cannot become a successful contribution.
                    if budget.remaining() <= 0 or budget.attempts >= budget.max_attempts:
                        outcome = "time_limit" if budget.remaining() <= 0 else "call_limit"
                        break
                    if e.kind == "model_failure":
                        outcome, error = "inference_failure", str(e)
                        break
                if tracker.agreed:
                    outcome = "agreement"
                    break
                if tracker.stalled:
                    outcome = "disagreement_stall"
                    break
    except KeyboardInterrupt:
        outcome, error = "cancelled", "Interrupted by user"
    except Exception as e:
        from llm_broker.budget import BudgetExceeded
        outcome = e.reason if isinstance(e, BudgetExceeded) else "internal_failure"
        error = str(e)
        writer.event("trial_error", trial_id=trial_id, error=error, outcome=outcome)
    if failed and not succeeded and outcome == "turn_limit":
        outcome = "inference_failure"
    usage = broker.trial_usage(trial_id)
    failed = attempted - succeeded  # includes interrupted or expired in-flight turns
    return TrialOutcome(trial_id=trial_id, task_id=task.task_id, seed=seed, mode=mode,
        outcome=outcome, final_answer=tracker.active.prediction if tracker.active else None,
        provisional=not tracker.agreed, answer_agreement=tracker.agreed, agreeing_agents=sorted(tracker.votes),
        grading="unscored", correct=None, turns_attempted=attempted, turns_succeeded=succeeded, turns_failed=failed,
        usage={**usage.model_dump(), "total_tokens": usage.total_tokens},
        duration_seconds=round(time.monotonic() - budget.started, 3), error=error)


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=("ollama", "mock"), default="ollama")
    ap.add_argument("--model", default=None)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--references", type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--max-turns", type=int, default=9)
    ap.add_argument("--max-calls", type=int, default=100)
    ap.add_argument("--timeout-s", type=float, default=1200)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--require-gpu", action="store_true", help="require GPU detection AND model fully on GPU")
    ap.add_argument("--out", type=Path, required=True)
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        tasks, refs = load_inputs(args.tasks, args.references)
        if args.limit is not None:
            if args.limit <= 0:
                raise ValueError("--limit must be positive")
            tasks = tasks[:args.limit]
        settings = InferenceSettings.from_env(backend=args.backend, model=args.model, max_concurrency=1)
        result = run_prototype(tasks, refs, args.out, settings=settings, seeds=args.seeds,
            max_turns=args.max_turns, max_calls=args.max_calls, timeout_s=args.timeout_s, require_gpu=args.require_gpu)
    except (ValueError, OSError) as e:
        print(f"Cannot start prototype: {e}", file=sys.stderr)
        return 1
    print(f"Artifacts: {args.out.resolve()}\nRun status: {result['manifest']['status']}")
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
