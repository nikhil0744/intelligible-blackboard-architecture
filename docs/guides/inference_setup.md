# Inference setup (Student 2)

How to run the study model on a free Colab or Kaggle GPU, check that the environment is fit
to run trials, and produce the real-model validation artifact. Notebook template:
[`notebooks/s2_inference_setup.ipynb`](../../notebooks/s2_inference_setup.ipynb).

## Frozen pilot settings

| Setting | Value | Variable |
|---|---|---|
| Model | `qwen2.5:7b-instruct-q4_K_M` (one shared model) | `LLM_MODEL` |
| Temperature | 0.3 | `LLM_TEMPERATURE` |
| Context window | 8,192 tokens | `LLM_NUM_CTX` |
| Maximum response | 768 tokens | `LLM_MAX_TOKENS` |
| Repair responses after an invalid first response | 2 | `LLM_MAX_REPAIRS` |
| Request timeout | 120 s | `LLM_TIMEOUT_S` |
| Concurrent requests | 1 | `LLM_MAX_CONCURRENCY` |

These live in `llm_broker.InferenceSettings`. Change them only during development and pilot
calibration, then freeze them. The effective values are written into every run manifest.

There is no fallback: if the model is missing or the backend is down, the call fails. Nothing
pulls a model, switches model, or moves to a paid API automatically.

## Server rules

- Bind Ollama to the notebook only: `OLLAMA_HOST=127.0.0.1:11434`.
- One loaded model, one request at a time: `OLLAMA_MAX_LOADED_MODELS=1`, `OLLAMA_NUM_PARALLEL=1`.
- Free GPUs vary between sessions. Do not assume a T4; the preflight records what was actually allocated.
- Runtime-local files are lost when the session ends. Write artifacts to Drive (Colab) or
  `/kaggle/working` (Kaggle).

## Commands

```bash
python -m scripts.s2_preflight                    # exit 1 if a required check fails
python -m scripts.s2_smoke --domain medical       # real model, stub board
python -m scripts.s2_smoke --task-file task.json  # a real benchmark task prompt
python -m scripts.s2_smoke --scheduler            # same agents through the integrated scheduler
python -m scripts.s2_smoke --backend mock         # offline check of the pipeline itself
```

The preflight checks: GPU and memory, backend health, that the model is pulled (with its
digest), that the context window is supported and matches the backend, a real probe call
(with cold-load time), whether token usage is reported, and whether the model is fully in GPU memory.

## Validation artifact

Each smoke run writes `outputs/s2_smoke/<run-id>/` (or `--out`):

| File | Content |
|---|---|
| `manifest.json` | settings, model digest, agent specifications, task, preflight report, git revision |
| `calls.jsonl` | one line per model call: trial, phase, repair flag, tokens, latency, attempts, error |
| `raw_responses.jsonl` | every raw model response and, if rejected, the reason |
| `contributions.jsonl` | committed contributions |
| `failures.jsonl` | turns that produced no valid contribution |
| `summary.json` | repair statistics, failure kinds, usage, latency, prompt-token estimate accuracy |

`outputs/` is ignored by git; do not commit run artifacts.

## Accounting for other students

- Wrap each trial: `with usage_scope(trial_id=..., phase=...)`. Phases: `deliberation`,
  `baseline_replay`, `candidate_replay`, `tool`, `evaluator`.
- Per-trial cost: `broker.trial_usage(trial_id)`. `broker.usage()` is the batch-wide running total.
- Unknown usage is `None`, never 0. `UsageSummary.total_tokens` is `None` if any call is unknown.
- One broker serves the whole batch; `close()` only releases threads.
- Validate replayed outputs with `prompts.validate_decision`, the same function ordinary turns use.
