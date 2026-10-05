# Intelligible Blackboard Architecture with Counterfactual Agents

A multi-agent research project combining prediction/explanation exchanges (PXP), shared blackboard state, deterministic scheduling, counterfactual deadlock recovery, benchmark evaluation, and a React visualization dashboard.

## Repository layout

| Location | Purpose |
| --- | --- |
| `src/contracts/` | Shared Pydantic models and exported JSON schemas |
| `src/blackboard/`, `src/storage/`, `src/scheduler/` | Blackboard state, Redis-backed storage, scheduling and concurrency |
| `src/agents/`, `src/llm_broker/`, `src/prompts/` | Persona agents, Ollama/LiteLLM/mock inference and prompt templates |
| `src/counterfactual/`, `src/sandbox/` | Attribution, isolated rollouts, scoring and recovery |
| `src/ingest/`, `src/benchmarks/`, `src/analytics/` | Dataset adapters, batch trials, metrics and plots |
| `src/streaming/` | FastAPI HTTP and WebSocket service |
| `src/fixtures/`, `src/scripts/` | Shared mocks, sample deadlock data and runnable smoke checks |
| `frontend/` | React/Vite dashboard |
| `tests/` | Offline unit and integration tests |
| `docs/` | Contributor guides, research references and reviews |
| `examples/benchmark-results/` | Preserved example simulation results and figures |
| `outputs/` | New generated results and figures; ignored by Git |

Module names remain unchanged: for example, `from contracts.schemas import TrialConfig` still works after installation. The source directories have moved under `src/`; install the project before running Python commands.

## Setup

Use Python 3.11 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
cp .env.example .env
```

On Windows, activate with `.venv\Scripts\activate`. `pip install -r requirements.txt` is an equivalent development installation. All Python dependencies are defined in `pyproject.toml`.

Ollama and the offline mock backend do not require LiteLLM. To use hosted models or other LiteLLM-supported servers:

```bash
python -m pip install -e ".[dev,litellm]"
```

Configure model/backend settings using `.env.example`. Live Ollama runs require a running Ollama server and the selected model pulled locally. Redis-backed deployments require a running Redis service; offline tests use in-memory clients.

## Run the backend and dashboard

**Known prototype error:** live runs can still exhaust the three validation
attempts when models write ambiguously grouped fractions, such as
`1/3 / 19/30` instead of `(1/3) / (19/30)`. A run may therefore stop with
`arithmetic_mismatch` before producing a valid contribution. The prototype
still needs live-model troubleshooting before it can be relied on for a demo.

For the presentation prototype, use [the prototype quickstart](docs/guides/prototype_quickstart.md)
and [the Colab T4 notebook](notebooks/prototype_colab.ipynb). The supported entrypoints are
`python -m scripts.prototype_run` for compute and `python -m scripts.prototype_demo`
for a live artifact viewer. This path records actual questions, explicit three-agent
answer agreement, separately graded answers, per-trial usage, and replayable traces.
Recovery is disabled. The legacy batch runner and the React dashboard's sample metrics
are not the presentation compute/result path.

For harder questions with a counterfactual intervention after observed live
disagreement, use the separate [natural recovery demo](docs/guides/prototype_recovery_demo.md)
entrypoint, `scripts.prototype_recovery`. It verifies an isolated historical
alternative before promotion and requires new live endorsements afterward.
Both prototype runners check declared numerical calculations with an exact
fraction calculator and repair stale endorsements before accepting them. This
checks arithmetic consistency; correctness against the task is graded separately.

```bash
uvicorn streaming.main:app --reload --port 8000
```

API documentation: <http://localhost:8000/docs>. The streaming service exposes health, session history, event ingestion and WebSocket endpoints. Producers must feed events into the service; starting it alone does not connect a separate reasoning process automatically.

In another terminal, with Node.js and npm installed:

```bash
cd frontend
npm ci
npm run dev
```

Open <http://localhost:5173>. The dashboard includes demo data. Build it with `npm run build` from `frontend/`.

## Verification and examples

```bash
python -m pytest -q
python -m scripts.s2_smoke --backend mock --parallel
python -m fixtures.spike_starter
python -m benchmarks.mscore --mock --trials 8
```

The benchmark command writes `outputs/results.csv`, `outputs/results.json`, and `outputs/figures/`. Override destinations using `--out-csv`, `--out-json`, and `--figures-dir`. The plotting helper also defaults to `outputs/figures/`.

Saved [example results](examples/benchmark-results/README.md) are simulation outputs. They illustrate output formats and plots rather than establish live-model performance.

To regenerate tracked contract schemas intentionally:

```bash
python -m contracts.export
```

This writes `src/contracts/stubs/`. CI instead exports to a temporary directory and checks that the committed schemas match the models.

## Further reading

- [Agent engineering guide](docs/guides/student2.md)
- [Streaming, dashboard and benchmarking guide](docs/guides/student4.md)
- [Research and benchmark references](docs/references/README.md)

The contributor guides preserve historical integration notes; this README is the current setup and layout reference.
