# Local inference and agent validation

The benchmark workflow targets one shared `qwen2.5:7b-instruct-q4_K_M` model with three personas and sequential calls. `llm_broker.InferenceSettings` reads environment settings and validates resource limits. It never changes backend/model or downloads a model silently.

| Variable | Default |
| --- | --- |
| `LLM_BACKEND` | `ollama` |
| `LLM_MODEL` | `qwen2.5:7b-instruct-q4_K_M` |
| `LLM_TEMPERATURE` | `0.3` |
| `LLM_NUM_CTX` | `8192` |
| `LLM_MAX_TOKENS` | `768` |
| `LLM_MAX_REPAIRS` | `2` |
| `LLM_MAX_RETRIES` | `2` |
| `LLM_TIMEOUT_S` | `120` |
| `LLM_MAX_CONCURRENCY` | `1` |
| `OLLAMA_HOST` | `http://localhost:11434` |

`PEXAgent` validates parsed output against the board. A contradictory RATIFY or missing target causes a bounded repair request, not invented agreement. Agent roles remain unchanged when counterfactual capability is enabled. The in-memory board and scheduler share zero-based turn numbering through an explicit adapter.

Use `usage_scope(trial_id=..., phase=...)` around task inference and `broker.trial_usage(trial_id)` for task cost. The broker lifetime total is not a task total. `None` denotes unknown usage; successful retries retain unknown earlier-attempt usage. Prompt lengths are estimates, with truncation reported in contribution metadata, not exact tokenizer measurements.

The inference, semantic-validation, prompt-budget, and accounting foundations were selectively recovered from archived commit `79345bb` (`archive/2026-11-13/s3-compute-readiness`, including the S2 commits). They were adapted and tested independently; the archived prototype runner, arithmetic-specific policy, and simulated benchmark environments were not imported.

Offline tests exercise actual agent classes with explicitly scripted model backends. They do not establish live-model quality, GPU readiness, or official benchmark performance.
