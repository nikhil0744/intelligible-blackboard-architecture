"""Student 2 smoke test: a real local model debating on the stub blackboard.

    python -m scripts.s2_smoke                       # uses .env (default: Ollama + qwen2.5 7B)
    python -m scripts.s2_smoke --backend mock        # offline
    python -m scripts.s2_smoke --domain medical --turns 6 --parallel

Checks: Ollama reachable, model pulled, every turn yields schema-valid PEX JSON.
"""

from __future__ import annotations

import argparse
import sys
import time

from agents import InMemoryBoard, act_parallel, build_panel
from llm_broker import build_broker_from_env
from llm_broker.types import LLMError
from prompts import render_board

TASKS = {
    "general": ("demo-gen", "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"),
    "medical": ("demo-med", "A 64-year-old with type 2 diabetes has eGFR 28 mL/min/1.73m2 and is on metformin 1000 mg BID. What should be done with the metformin order?"),
    "engineering": ("demo-eng", "An EV battery pack shows accelerated capacity fade only in units assembled on one production line. Which lifecycle stage most likely introduced the defect and what should be checked first?"),
    "data": ("demo-data", "Given monthly CSVs where some files record temperature in Fahrenheit and others in Celsius, what pipeline step is needed before computing the annual mean temperature?"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=None, help="ollama | litellm | mock (overrides .env)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--domain", default="general", choices=sorted(TASKS))
    ap.add_argument("--agents", type=int, default=3)
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--parallel", action="store_true", help="also time one parallel round")
    args = ap.parse_args()

    overrides = {k: v for k, v in {"llm_backend": args.backend, "llm_model": args.model}.items() if v}
    broker = build_broker_from_env(**overrides)
    be = broker.backend
    print(f"backend={be.name} model={broker.default_model} max_concurrency={broker.max_concurrency}")
    if be.name == "ollama":
        try:
            models = be.list_models()
        except LLMError as e:
            print(f"FAIL: {e}\n -> start Ollama (`ollama serve`) and retry.")
            return 1
        if broker.default_model not in models:
            print(f"FAIL: model {broker.default_model!r} not pulled. Have: {models}\n -> ollama pull {broker.default_model}")
            return 1

    board = InMemoryBoard()
    task_id, question = TASKS[args.domain]
    sid = board.create_session(task_id, question).session_id
    panel = build_panel(broker, domain=args.domain, size=args.agents, counterfactual_density=0.33)
    print("panel:", panel)

    with broker:
        for t in range(args.turns):
            agent = panel[t % len(panel)]
            t0 = time.perf_counter()
            c = agent.step(board, sid)
            print(f"turn {t}: {agent.agent_id:<22} {c.tag.value:<7} conf={c.payload.prediction.confidence:.2f} "
                  f"tok={c.token_usage} {time.perf_counter() - t0:5.1f}s | {c.payload.prediction.claim[:80]}")
        if args.parallel:
            t0 = time.perf_counter()
            act_parallel(panel, board.get_snapshot(sid))
            print(f"parallel round of {len(panel)} agents: {time.perf_counter() - t0:.1f}s")
        print("\n" + render_board(board.get_snapshot(sid)))
        u = broker.usage()
        print(f"\nusage: calls={u.calls} failures={u.failures} tokens={u.total_tokens} "
              f"avg_latency={u.total_latency_ms / max(u.calls, 1):.0f}ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
