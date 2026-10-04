"""Student 2 real-model validation run (tasks S2.4 / S2.6).

    python -m scripts.s2_smoke                         # .env settings (default: Ollama + qwen2.5 7B)
    python -m scripts.s2_smoke --backend mock          # offline
    python -m scripts.s2_smoke --domain medical --turns 6
    python -m scripts.s2_smoke --task-file task.json   # a real task: {"task_id", "question", "context"?}
    python -m scripts.s2_smoke --scheduler             # the same agents driven by the integrated scheduler

Every run writes a validation artifact to outputs/s2_smoke/<run-id>/ (or --out):
    manifest.json        settings, model digest, agent specifications, task, preflight, git revision
    calls.jsonl          one line per model call (usage ledger)
    raw_responses.jsonl  every raw model response, with the reason it was rejected if it was
    contributions.jsonl  the contributions that were committed
    failures.jsonl       turns that produced no valid contribution
    summary.json         repair statistics, failures, usage, latency

Exit code: 0 = ran; 1 = preflight failed; 2 = no turn produced a valid contribution.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents import AgentTurnError, InMemoryBoard, build_panel, panel_manifest, register_panel
from llm_broker import InferenceSettings, build_broker, usage_scope
from llm_broker.preflight import format_report, run_preflight
from prompts import render_board

MANIFEST_VERSION = 1

TASKS = {
    "general": ("demo-gen", "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"),
    "medical": ("demo-med", "A 64-year-old with type 2 diabetes has eGFR 28 mL/min/1.73m2 and is on metformin 1000 mg BID. What should be done with the metformin order?"),
    "engineering": ("demo-eng", "An EV battery pack shows accelerated capacity fade only in units assembled on one production line. Which lifecycle stage most likely introduced the defect and what should be checked first?"),
    "data": ("demo-data", "Given monthly CSVs where some files record temperature in Fahrenheit and others in Celsius, what pipeline step is needed before computing the annual mean temperature?"),
}


def git_revision() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def load_task(args: argparse.Namespace) -> Dict[str, Any]:
    if args.task_file:
        data = json.loads(Path(args.task_file).read_text(encoding="utf-8"))
        return {"task_id": str(data["task_id"]), "question": data["question"], "context": data.get("context") or {},
                "source": str(args.task_file)}
    task_id, question = TASKS[args.domain]
    return {"task_id": task_id, "question": question, "context": {}, "source": "built-in demo task"}


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _raw_rows(turn: int, agent_id: str, raw: List[str], reasons: List[str], accepted: bool) -> List[Dict[str, Any]]:
    """One row per model response. Response i was rejected for reasons[i]; the last one is accepted if the turn succeeded."""
    rows = []
    for i, text in enumerate(raw):
        ok = accepted and i == len(raw) - 1
        rows.append({"turn": turn, "agent_id": agent_id, "attempt": i + 1, "accepted": ok,
                     "rejected_because": None if ok else (reasons[i] if i < len(reasons) else None), "raw": text})
    return rows


def run_smoke(args: argparse.Namespace) -> Dict[str, Any]:
    settings = InferenceSettings.from_env(backend=args.backend, model=args.model)
    broker = build_broker(settings)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) if args.out else Path("outputs") / "s2_smoke" / run_id
    out.mkdir(parents=True, exist_ok=True)

    preflight = run_preflight(broker, settings, probe=not args.no_probe)
    print(format_report(preflight))

    task = load_task(args)
    panel = build_panel(broker, domain=args.domain, size=args.agents, counterfactual_count=args.capable,
                        seed=args.seed, **settings.agent_kwargs())
    mode = "scheduler" if args.scheduler else "direct"
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "kind": "s2_real_model_validation",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "mode": mode,
        "backend": broker.backend.name,
        "model": broker.default_model,
        "model_digest": preflight.get("model_digest"),
        "settings": settings.to_dict(),
        "seed": args.seed,
        "max_turns": args.turns,
        "task": task,
        "agents": panel_manifest(panel),
        "preflight": preflight,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    if not preflight["ok"]:
        return {"exit_code": 1, "out_dir": str(out), "manifest": manifest, "summary": None}

    broker.ledger.attach_file(out / "calls.jsonl")
    raw_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    contributions: List[Any] = []

    def record_ok(turn: int, c: Any) -> None:
        contributions.append(c)
        raw_rows.extend(_raw_rows(turn, c.agent_id, c.metadata.get("raw_responses", []), c.metadata.get("repair_reasons", []), True))
        print(f"turn {turn}: {c.agent_id:<22} {c.tag.value:<7} conf={c.payload.prediction.confidence:.2f} "
              f"tok={c.token_usage} attempts={c.metadata.get('parse_attempts')} | {c.payload.prediction.claim[:80]}")

    def record_failure(turn: int, e: AgentTurnError) -> None:
        failures.append({"turn": turn, **e.to_record()})
        raw_rows.extend(_raw_rows(turn, e.agent_id, e.raw_responses, e.repair_reasons, False))
        print(f"turn {turn}: {e.agent_id:<22} FAILED ({e.kind}): {str(e)[:120]}")

    t_start = time.perf_counter()
    with usage_scope(trial_id=run_id, phase="deliberation"):
        if args.scheduler:
            from scheduler.engine import DeterministicScheduler
            from scheduler.policies import RoundRobinPolicy
            from storage.redis_store import InMemoryRedisClient, RedisBlackboardStore

            store = RedisBlackboardStore(redis_client=InMemoryRedisClient())
            sid = f"s2_smoke_{run_id}"
            store.create_session(sid, task["task_id"], task["question"], initial_context=task["context"])
            sched = DeterministicScheduler(session_id=sid, store=store, policy=RoundRobinPolicy(),
                                           consensus_threshold=len(panel))
            register_panel(sched, panel)
            for turn in range(args.turns):
                if store.get_snapshot(sid).status.value != "ACTIVE" or sched.queue.is_paused:
                    break
                try:
                    c = sched.step()
                except AgentTurnError as e:
                    record_failure(turn, e)
                    sched.active_agent_id = None
                    continue
                if c is None:
                    break
                record_ok(turn, c)
            final = store.get_snapshot(sid)
        else:
            board = InMemoryBoard()
            sid = board.create_session(task["task_id"], task["question"], task["context"]).session_id
            for turn in range(args.turns):
                try:
                    record_ok(turn, panel[turn % len(panel)].step(board, sid))
                except AgentTurnError as e:
                    record_failure(turn, e)
            final = board.get_snapshot(sid)
    wall_s = time.perf_counter() - t_start
    broker.close()

    usage = broker.trial_usage(run_id)
    calls = broker.ledger.records(trial_id=run_id)
    lat = [r.latency_ms for r in calls if r.ok]
    ratios = [r.prompt_tokens / r.estimated_prompt_tokens for r in calls
              if r.prompt_tokens and r.estimated_prompt_tokens]
    reason_codes = Counter(reason.split(":", 1)[0] for row in failures for reason in row["repair_reasons"])
    reason_codes.update(reason.split(":", 1)[0] for c in contributions for reason in c.metadata.get("repair_reasons", []))
    turns_run = len(contributions) + len(failures)
    summary = {
        "run_id": run_id,
        "mode": mode,
        "final_status": final.status.value,
        "turns_attempted": turns_run,
        "turns_succeeded": len(contributions),
        "turns_failed": len(failures),
        "failure_kinds": dict(Counter(f["kind"] for f in failures)),
        "repair": {
            "turns_valid_first_try": sum(1 for c in contributions if c.metadata.get("parse_attempts") == 1),
            "turns_valid_after_repair": sum(1 for c in contributions if (c.metadata.get("parse_attempts") or 1) > 1),
            "repair_calls": usage.repairs,
            "rejection_reasons": dict(reason_codes),
            "normalizations": dict(Counter(n for c in contributions for n in c.metadata.get("normalizations", []))),
        },
        "tags": dict(Counter(c.tag.value for c in contributions)),
        "usage": {**usage.model_dump(), "total_tokens": usage.total_tokens, "known_total_tokens": usage.known_total_tokens},
        "latency_ms": {
            "per_call_mean": round(statistics.fmean(lat), 1) if lat else None,
            "per_call_median": round(statistics.median(lat), 1) if lat else None,
            "per_call_max": round(max(lat), 1) if lat else None,
            "wall_seconds": round(wall_s, 2),
            "model_load_probe_seconds": (preflight.get("probe") or {}).get("seconds_including_model_load"),
        },
        "prompt_tokens_actual_over_estimate": round(statistics.fmean(ratios), 3) if ratios else None,
        "prompts_truncated": sum(1 for c in contributions if (c.metadata.get("prompt_budget") or {}).get("omitted_contribution_ids")),
    }
    _write_jsonl(out / "raw_responses.jsonl", raw_rows)
    _write_jsonl(out / "failures.jsonl", failures)
    _write_jsonl(out / "contributions.jsonl", [c.model_dump(mode="json") for c in contributions])
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print("\n" + render_board(final))
    print(f"\nturns ok={summary['turns_succeeded']} failed={summary['turns_failed']} repair_calls={usage.repairs} "
          f"tokens={usage.total_tokens if usage.total_tokens is not None else 'unknown'} "
          f"mean_call={summary['latency_ms']['per_call_mean']}ms\nartifact: {out}")
    return {"exit_code": 0 if contributions else 2, "out_dir": str(out), "manifest": manifest, "summary": summary}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Student 2 real-model validation run")
    ap.add_argument("--backend", default=None, help="ollama | litellm | mock (overrides .env)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--domain", default="general", choices=sorted(TASKS))
    ap.add_argument("--task-file", default=None, help='JSON file: {"task_id": ..., "question": ..., "context": {...}}')
    ap.add_argument("--agents", type=int, default=3)
    ap.add_argument("--capable", type=int, default=1, help="number of counterfactual-capable agents (0..agents)")
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scheduler", action="store_true", help="drive the agents through the integrated scheduler")
    ap.add_argument("--out", default=None, help="artifact directory (default outputs/s2_smoke/<run-id>)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--no-probe", action="store_true", help="skip the preflight model call")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    return run_smoke(build_parser().parse_args(argv))["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
