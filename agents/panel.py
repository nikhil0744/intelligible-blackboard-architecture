"""Build agent panels per domain and run agents in parallel (non-blocking)."""

from __future__ import annotations

import asyncio
import math
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence

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
    **agent_kwargs,
) -> List[PEXAgent]:
    """Create `size` agents cycling through the domain's personas (+ general critic as filler).

    `counterfactual_density` (0, .33, .66, 1.0) marks round(size*density) agents as
    counterfactual-capable, chosen with `seed` for reproducible ablations.
    `models` optionally assigns different base models round-robin (model-diversity study).
    """
    personas = list_personas(domain) or list_personas("general")
    if len(personas) < size:
        personas += [p for p in list_personas("general") if p not in personas]
    rng = random.Random(seed)
    n_cf = int(math.floor(size * counterfactual_density + 0.5))
    cf_idx = set(rng.sample(range(size), n_cf))
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
        futures = [ex.submit(a.act, snapshot) for a in agents]
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
