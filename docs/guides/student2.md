> Historical contributor guide. Run setup commands from the repository root; see the [current README](../../README.md) for the source layout and setup.

# Student 2 — Agent Engineering & Model Inference Pipeline

Branch: `student-2-agents` (built on `student-1-core` contracts).

## What's here
| Path | Deliverable (plan §3) |
|---|---|
| `src/llm_broker/` | Multi-threaded local model calling interface + hardware-resource broker |
| `src/llm_broker/backends/` | `OllamaBackend` (stdlib HTTP, schema-constrained JSON), `LiteLLMBackend` (vLLM/hosted), `MockBackend` (offline) |
| `src/prompts/` | Prompt matrix: `personas.json` (general / medical / engineering / data), PXP rules, PEX JSON schema, robust parser |
| `src/agents/` | `PEXAgent` lifecycle, `build_panel` (ablation density), `act_parallel`, `BoardClient` interface + `InMemoryBoard` stub |
| `tests/test_s2_agents.py` | 20 offline tests (no GPU needed) |
| `src/scripts/s2_smoke.py` | Live check against a real local model |

## Setup (plan §7, Student 2)
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev,litellm]"                    # litellm optional
cp .env.example .env
ollama pull qwen2.5:7b-instruct-q4_K_M
pytest -q                                             # offline
python -m scripts.s2_smoke --domain medical --turns 4 --parallel   # live
```

## Agent lifecycle (`PEXAgent.act`)
1. **read** – render snapshot (task, context, last 12 posts with ids) + persona system prompt.
2. **formulate** – broker call with JSON-schema-constrained decoding; on invalid JSON, re-prompt with the error (≤2 repairs).
3. **select tag** – enforce PXP rules: empty board ⇒ `REVISE`/no target; invalid/truncated target ⇒ repaired to a real id; `RATIFY` inherits the target's claim.
4. **submit** – build S1 `AgentContribution` (`turn_index`, `token_usage`, `latency_ms`, persona/model metadata). `step()` submits and re-reads once if the board advanced.

## Integration contracts
- **S1 (scheduler/store):** implement `agents.BoardClient` → `get_snapshot(session_id)`, `submit(contribution)` (raise `ValueError` containing `turn_index` on stale turns). Async loop: `await agents.aact(agent, snapshot)`; the scheduler owns commit order.
- **S3 (sandbox):** reuse the *same* `ModelBroker`. Replay with `agent.act(modified_snapshot, extra_instructions=..., is_counterfactual=True)`; persist a prompt change via `agent.system_addendum`. Agents are stateless between turns, so deep-copied history replays are exact inputs.
- **S4 (analytics):** tokens per contribution in `token_usage`; totals via `broker.usage()` / `usage_by_agent()`; ablation panels via `build_panel(..., counterfactual_density=0.33, seed=...)` or `build_panel_for_benchmark(broker, BenchmarkType.MSCORE)`.

## Notes / open items for the team
- `AgentRole.COUNTERFACTUAL` is used as the agent's role when counterfactual-capable (persona role kept in `metadata.persona`). Confirm with S1 this is the intended use.
- Parallel agents all get `turn_index = len(snapshot.contributions)`; S1 must re-index when committing a batch.
- CPU-only laptops: keep `LLM_MAX_CONCURRENCY=1–2`; Ollama also needs `OLLAMA_NUM_PARALLEL` ≥ that value to truly serve in parallel.
- Model-diversity study: `build_panel(..., models=["qwen2.5:7b-instruct-q4_K_M", "llama3.1:8b", "mistral:7b"])`.

## Checklist (Milestone 2 target, weeks 4–5)
- [x] Broker: concurrency cap, retries, async, token accounting
- [x] Ollama + LiteLLM + Mock backends
- [x] Persona matrix for all 3 benchmark domains + general
- [x] PEX JSON schema + tolerant parser + repair loop
- [x] PEXAgent read → formulate → tag → submit
- [x] Offline test suite passing
- [ ] Run `src/scripts/s2_smoke.py` on your machine with the real 7B model; record JSON-validity rate
- [ ] Swap `InMemoryBoard` for S1's store once `storage/` / `scheduler/` land
- [ ] Hand S3 the broker + `act(..., is_counterfactual=True)` API
