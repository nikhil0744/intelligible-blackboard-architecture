"""Explain whether the counterfactual resolver ran in a prototype run, and why.

python -m scripts.prototype_status --run-dir outputs/presentation/natural-recovery-001
"""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.prototype_artifacts import read_state
from scripts.prototype_recovery import RESOLVER

PHASES = {
    "ordinary_discussion": "ordinary discussion on the live board",
    "sandbox_alternative": "counterfactual resolver writing an alternative in the sandbox",
    "sandbox_peer_check": "peers checking the resolver's alternative in the sandbox",
    "sandbox_peer_check_independent_check": "peers calculating privately before the sandbox check",
    "live_confirmation": "peers confirming the promoted proposal on the live board",
    "live_confirmation_independent_check": "peers calculating privately before live confirmation",
}


def describe(run_dir: Path) -> list[str]:
    state = read_state(Path(run_dir))
    manifest, events, results = state["manifest"], state["events"], state["results"]
    if not manifest:
        return [f"No manifest in {run_dir}; the run has not started."]
    if manifest.get("kind") != "natural_counterfactual_demo":
        return ["RECOVERY DISABLED: this is the ordinary prototype (RECOVERY_DEMO = False).",
                "The counterfactual resolver and sandbox cannot run in this mode."]
    posts = [e for e in events if e["type"] == "contribution"]
    ordinary = [e for e in posts if e.get("phase") == "ordinary_discussion"]
    resolver_posts = [e for e in posts if e["contribution"]["agent_id"] == RESOLVER and e.get("phase") == "sandbox_alternative"]
    lines = [f"Task: {manifest['tasks'][0]['task_id']}  status: {manifest.get('status')}  "
             f"trigger policy: {manifest.get('recovery_trigger')}",
             f"Ordinary turns posted: {len(ordinary)}; resolver sandbox posts: {len(resolver_posts)}"]
    report = manifest.get("recovery")
    if report is None:
        started = [e for e in events if e["type"] == "turn_started"]
        if not started:
            return lines + ["Waiting for preflight/model load; no agent turn has started."]
        phase = started[-1].get("phase", "")
        entered = any(e["type"] == "recovery_stage" for e in events)
        return lines + [f"Running: {PHASES.get(phase, phase)}.",
                        "Resolver has been spawned." if entered else "No conflict yet; resolver not spawned."]
    result = results[-1] if results else {}
    outcome = result.get("outcome")
    if not report["triggered"]:
        if outcome == "ordinary_agreement":
            why = ("All three agents ratified the same answer without a conflict, so the resolver "
                   "correctly stayed inactive. Try another task or seed.")
        elif outcome == "ordinary_turn_limit":
            why = (f"No qualifying conflict within {manifest['limits']['ordinary_turns']} ordinary turns "
                   "(a different agent must REFUTE/REJECT the active proposal or REVISE it to a different answer).")
        else:
            why = f"The run ended before any conflict: {outcome or manifest.get('status')}. {manifest.get('error') or ''}".strip()
        return lines + ["Resolver NOT spawned. " + why] + _answer(result)
    lines += [f"Conflict observed: {report['trigger_observed']}; sandbox rewound before turn {report['rollback_before_turn']}.",
              f"Resolver spawned; its sandbox proposal: {resolver_posts[0]['contribution']['payload']['prediction']['claim']!r}"
              if resolver_posts else "Resolver spawned but did not post (see the turn_failed event in trace.jsonl).",
              f"Sandbox isolated from live board: {report.get('sandbox_isolated')}",
              f"Sandbox approved by all three peers: {report['sandbox_approved']} "
              f"(final sandbox answer {report['sandbox_final_answer']!r}, author {report['sandbox_approved_agent_id']})",
              f"Promoted to live board: {report['promoted']}; live recovery confirmed: {report['recovery_confirmed']}"]
    return lines + _answer(result)


def _answer(result: dict) -> list[str]:
    if not result:
        return []
    return [f"Outcome: {result['outcome']}; final answer {result['final_answer']!r}; "
            f"graded {result['grading']}, correct={result['correct']}"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print("\n".join(describe(args.run_dir)))
    except (OSError, ValueError, KeyError) as e:
        print(f"Cannot read run status: {type(e).__name__}: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
