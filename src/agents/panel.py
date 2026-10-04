"""Build agent panels per domain and run agents in parallel (non-blocking)."""

from __future__ import annotations

import asyncio
import contextvars
import math
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence

from contracts.schemas import AgentContribution, BenchmarkType, BlackboardSnapshot
from llm_broker import ModelBroker
from prompts import list_personas

from .base import AgentTurnError, PEXAgent

BENCHMARK_DOMAIN: Dict[BenchmarkType, str] = {
    BenchmarkType.MEDAGENTBENCH: "medical",
    BenchmarkType.MSCORE: "engineering",
    BenchmarkType.KRAMABENCH: "data",
}


def build_panel(
    broker: ModelBroker,
    domain: str = "general",
    size: int = 3,
    counterfactual_density: float = 0.0,
    models: Optional[Sequence[str]] = None,
    seed: int = 42,
    counterfactual_count: Optional[int] = None,
    **agent_kwargs,
) -> List[PEXAgent]:
    """Create `size` agents cycling through the domain's personas (+ general critic as filler).

    The roster (agent ids, personas, roles, models, decoding settings, per-agent seeds) depends
    only on `domain`, `size`, `models`, `seed` and `agent_kwargs`, never on the capability
    condition, so matched conditions differ in counterfactual capability alone.

    `counterfactual_count` (0..size) is the authoritative number of capable agents. If it is
    None, round(size * counterfactual_density) is used (0, .33, .66, 1.0 -> 0, 1, 2, 3 of 3).
    Capable agents are the first n of one seed-determined order, so the conditions are nested:
    the agent capable at n=1 is also capable at n=2 and n=3.
    `models` optionally assigns different base models round-robin (model-diversity study).
    """
    personas = list_personas(domain) or list_personas("general")
    if len(personas) < size:
        personas += [p for p in list_personas("general") if p not in personas]
    n_cf = counterfactual_count
    if n_cf is None:
        n_cf = int(math.floor(size * counterfactual_density + 0.5))
    if not 0 <= n_cf <= size:
        raise ValueError(f"counterfactual_count must be in 0..{size}, got {n_cf}")
    order = list(range(size))
    random.Random(seed).shuffle(order)
    cf_idx = set(order[:n_cf])
    panel = []
    for i in range(size):
        p = personas[i % len(personas)]
        panel.append(
            PEXAgent(
                agent_id=f"{p.id}_{i}",
                persona=p,
                broker=broker,
                model=models[i % len(models)] if models else None,
                counterfactual_capable=i in cf_idx,
                seed=seed + i,
                **agent_kwargs,
            )
        )
    return panel


def build_panel_for_benchmark(broker: ModelBroker, benchmark: BenchmarkType, **kw) -> List[PEXAgent]:
    return build_panel(broker, domain=BENCHMARK_DOMAIN[benchmark], **kw)


def register_panel(scheduler: Any, panel: Sequence[PEXAgent], priority: int = 1) -> None:
    """Register every agent as its own turn handler on S1's scheduler.

    Agents are registered under their persona role in every capability condition, so a
    counterfactual-capable agent keeps receiving ordinary turns.
    """
    for agent in panel:
        scheduler.register_agent(agent.agent_id, agent.role, turn_handler=agent, priority=priority)


def panel_manifest(panel: Sequence[PEXAgent]) -> List[Dict[str, Any]]:
    """Agent specifications for a run manifest."""
    return [a.spec() for a in panel]


def act_parallel(
    agents: Sequence[PEXAgent],
    snapshot: BlackboardSnapshot,
    max_workers: Optional[int] = None,
    return_exceptions: bool = False,
) -> List[AgentContribution | Exception]:
    """All agents read the SAME snapshot concurrently; results keep agent order.

    The scheduler (S1) decides which proposals to commit and in what order;
    each result has turn_index = len(snapshot.contributions), so S1 re-indexes on commit.
    """
    with ThreadPoolExecutor(max_workers=max_workers or len(agents) or 1) as ex:
        # copy_context: keep the caller's usage_scope (trial / phase) in the worker threads
        futures = [ex.submit(contextvars.copy_context().run, a.act, snapshot) for a in agents]
        out: List[AgentContribution | Exception] = []
        for f in futures:
            try:
                out.append(f.result())
            except AgentTurnError as e:
                if not return_exceptions:
                    raise
                out.append(e)
    return out


async def aact(agent: PEXAgent, snapshot: BlackboardSnapshot, **kw) -> AgentContribution:
    """Async wrapper for S1's asyncio event loop (runs inference off the loop thread)."""
    return await asyncio.to_thread(agent.act, snapshot, **kw)
