# Intelligible Blackboard Architecture with Counterfactual Agents

A multi-agent research project combining prediction/explanation exchanges (PXP), shared blackboard state, deterministic scheduling, counterfactual deadlock recovery, benchmark evaluation, and a React visualization dashboard.

## Run the benchmarks in Colab

| Notebook | What it runs |
| --- | --- |
| [KramaBench](https://colab.research.google.com/github/nikhil0744/intelligible-blackboard-architecture/blob/main/notebooks/colab_benchmarks.ipynb) | A fixed 20-task legal CSV profile and a three-task pilot. Includes bounded evidence recovery, Drive resume, and recorded playback. |
| [MSCoRe with recovery](https://colab.research.google.com/github/nikhil0744/intelligible-blackboard-architecture/blob/main/notebooks/colab_mscore_recovery.ipynb) | Three or 20 Chinese true/false construction judgments, with a recovery switch and a task picker for recorded playback. |
| [Original MSCoRe baseline](https://colab.research.google.com/github/nikhil0744/intelligible-blackboard-architecture/blob/main/notebooks/colab_mscore.ipynb) | Historical runner used for the three-task pilot and 20-task baseline. It stops on deadlock. |

Select a GPU runtime and run the notebook cells in order. The notebooks use local Ollama inference with a quantized Qwen2.5-7B model and save results to Google Drive. Each notebook pins its project checkout. In KramaBench, set `TASK_COUNT = 1` for a quick check or keep the default 20-task profile. In MSCoRe, select three tasks first. Use `RECOVERY_ENABLED` to toggle recovery and keep the model, task IDs, and runner version fixed when comparing two runs.

Read the task grades separately from consensus. MSCoRe v3.2.0 records independent first ballots, detects repeated split votes, and can open a new branch with a dissenting agent as the first proposer. A recovery answer replaces the original submission only after unanimous agreement. Cell 6 opens the recorded visualizer. The benchmark runs evaluate selected subsets. MedAgentBench remains unavailable in Colab pending verification of its official environment and evaluator.

The [KramaBench Colab guide](docs/guides/colab-benchmarks.md) covers profiles, resuming, and exporting results.

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
| `docs/` | Contributor guides and research references |
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

Open <http://localhost:5173>. The dashboard includes demo data and a mock ablation runner backed by the FastAPI benchmark endpoints. Use the Colab notebooks for evaluations on the selected benchmark tasks. Build it with `npm run build` from `frontend/`.

## Verification and examples

```bash
python -m pytest -q
python -m scripts.s2_smoke --backend mock --parallel
python -m fixtures.spike_starter
python -m benchmarks.mscore --mock --trials 8
```

The historical `benchmarks.mscore` simulation writes `outputs/results.csv`, `outputs/results.json`, and `outputs/figures/`. Override destinations using `--out-csv`, `--out-json`, and `--figures-dir`. The Colab MSCoRe notebooks use their own live runner. The `benchmarks.run` CLI provides the live KramaBench path.

Saved [example results](examples/benchmark-results/README.md) are simulation outputs. They illustrate output formats and plots rather than establish live-model performance.

To regenerate tracked contract schemas intentionally:

```bash
python -m contracts.export
```

This writes `src/contracts/stubs/`. CI instead exports to a temporary directory and checks that the committed schemas match the models.

## Further reading

- [Streaming, dashboard and benchmarking guide](docs/guides/student4.md)
- [Research and benchmark references](docs/references/README.md)

The contributor guides preserve historical integration notes; this README is the current setup and layout reference.
