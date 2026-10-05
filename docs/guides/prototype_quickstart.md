# Run compute and present the prototype

The supported presentation path is `scripts.prototype_run` plus its artifact viewer.
It uses the actual question, three sequential agents (reasoner, critic, verifier),
validated PXP posts, explicit answer agreement, separate reference grading, and
measured per-trial usage. Recovery is disabled. Redis and the React dashboard are
not required for this path.

For **harder questions with natural disagreement and a counterfactual agent**,
use the separate [recovery demo instructions](prototype_recovery_demo.md).
That optional runner starts with live agents, intervenes only after an observed
conflict, verifies an isolated alternative, and checks its promotion on the live
board. The ordinary compute commands below keep recovery disabled.

The implementation has offline regression checks. A real T4/Ollama run must still
pass on your Colab runtime: there is no GPU or Ollama server in the development
workspace. Model output is stochastic; agreement and correct answers are outcomes
to measure, not guaranteed by a working program.

## 1. Recommended order for your deadline

1. Upload [notebooks/prototype_colab.ipynb](../../notebooks/prototype_colab.ipynb)
   to Google Colab. Choose **Runtime → Change runtime type → T4 GPU**.
2. Run the notebook in order. When prompted, upload the prepared
   `outputs/blackboard_prototype_source.zip` from this repository.
3. Keep `LIMIT = 1`, `SEEDS = [42]`. Inspect preflight, final answer, trace,
   correctness, token counts, and runtime. Resolve any failed preflight first.
4. After the first run finishes, change `LIMIT = 3`. Rerun the start-compute,
   viewer, and progress cells. This creates a fresh three-question batch.
5. Download its results ZIP. Open `demo.html` on your laptop and rehearse it.
   Copy the generated PNGs and the measured table into your slides.
6. Only if time remains, change `LIMIT = 5` or add a second seed. These are
   additional runs of the same five questions, not new benchmark questions.

Do not start 100–200 trials for this deadline. Measure the first trial and reserve
at least 60–90 minutes for slides and rehearsal. Three completed, inspected trials
are enough to show the prototype's execution and instrumentation.

The notebook uploads the current source rather than cloning an old remote branch.
It checks every packaged file hash. Run manifests record the bundle digest, base
commit, dirty-source flag, model digest, effective settings, inputs, and seeds.
Your local changes do not need to be pushed to GitHub for this workflow.

If you edit files after the bundle was generated, regenerate it locally:

```bash
source .venv/bin/activate
python -m scripts.prototype_bundle
```

The bundle includes source, examples, tests, installation metadata, and these
instructions. It excludes `.git`, `.env`, virtual environments, results, and
credentials. Google Drive mounting happens in your notebook; output goes to a
unique folder under `MyDrive/blackboard_prototype/`.

## 2. Check the interface locally right now

From this repository's root, use the existing virtual environment. For a new
checkout, first install with `python -m pip install -e ".[dev]"` in a Python 3.11+
virtual environment.

```bash
source .venv/bin/activate
python -m scripts.prototype_run \
  --backend mock \
  --tasks examples/prototype/tasks.json \
  --references examples/prototype/references.json \
  --limit 3 --seeds 42 \
  --out outputs/presentation/local-check-001

python -m scripts.prototype_demo \
  --run-dir outputs/presentation/local-check-001
```

Open **http://127.0.0.1:8765**. Select a trial, move the turn slider, click Replay,
and inspect the results table. Stop the server with Ctrl+C. The downloaded or
generated `demo.html` also opens directly in a browser without a server.

**Mock mode is a scripted pipeline check.** Its fixed answer is deliberately not
a solution to your questions, and its usage is simulated. The viewer and charts
label it clearly. Use real Ollama results for empirical presentation claims.

Each output directory must be new: use `local-check-002` on a repeated run. This
prevents mixing old call records with new summaries. The runner refuses reuse.

## 3. Real compute command (used by the Colab notebook)

The notebook installs Ollama, starts a local server, pulls one model, sets these
environment variables, and runs preflight. You can use the same commands on a
machine with Ollama installed and an NVIDIA GPU:

```bash
export LLM_BACKEND=ollama
export LLM_MODEL=qwen2.5:7b-instruct-q4_K_M
export OLLAMA_HOST=http://127.0.0.1:11434
export LLM_TEMPERATURE=0.3
export LLM_NUM_CTX=8192
export LLM_MAX_TOKENS=768
export LLM_MAX_REPAIRS=2
export LLM_TIMEOUT_S=120
export LLM_MAX_CONCURRENCY=1
export LLM_MAX_RETRIES=2
```

In a separate terminal, start `ollama serve`. Then:

```bash
ollama pull qwen2.5:7b-instruct-q4_K_M
python -m scripts.s2_preflight --backend ollama

python -m scripts.prototype_run \
  --backend ollama \
  --tasks examples/prototype/tasks.json \
  --references examples/prototype/references.json \
  --limit 1 --seeds 42 \
  --max-turns 9 --max-calls 100 --timeout-s 1200 \
  --require-gpu \
  --out outputs/presentation/live-check-001
```

Inspect **gpu_available**, **probe_call**, and **model_fully_on_gpu** in preflight.
The last check uses Ollama's reported loaded-model GPU memory placement.
`--require-gpu` refuses compute unless both GPU detection and full model placement
pass. Removing that flag permits CPU inference, which is not the recommended
deadline workflow. There is no automatic fallback to mock mode or another model.

Limits apply per question/seed. The call ceiling includes repair requests and
transport retries. Request timeouts are capped by the remaining trial deadline.
Preflight/model-loading measurements and call records are separate from trial
token totals and runtime. A trial can end without agreement and still save its
provisional answer and an honest grading result.

Exit codes: `0` completed, `1` preflight/internal/export failure, `2` at least one
inference failure, `130` interrupted. Exit `0` can include turn/call/time limit
outcomes; inspect the table, not just the exit code.

## 4. Start the live demo while compute runs

On Colab, the notebook starts compute as a background process and serves the
viewer inside an iframe using Colab's `output.serve_kernel_port_as_iframe(8765,
height=850)`. Run the viewer cell after the run manifest appears. It polls the
saved records every 1.5 seconds and shows new contributions as they are committed.
The progress cell reports process status and the latest log without blocking.

Locally, start compute in one terminal and run this in another once
`manifest.json` exists:

```bash
source .venv/bin/activate
python -m scripts.prototype_demo \
  --run-dir outputs/presentation/live-check-001 \
  --port 8765
```

Open **http://127.0.0.1:8765**. The server reads only the selected run's artifacts.
It also works after compute finishes. **Follow latest** follows the selected
trial's new turns; select another trial in the dropdown to see it. Starting the
viewer does not start inference.

For tomorrow, keep a downloaded real-run `demo.html` ready as a saved replay.
You can demonstrate actual recorded reasoning even if Colab disconnects. Clearly
say whether you are showing a recorded run or newly executing inference.

## 5. What to inspect and use in slides

| File | What it contains |
| --- | --- |
| `manifest.json` | Source identity, model/settings, inputs, seeds, preflight and run status |
| `trace.jsonl` | Incrementally saved raw model responses/prompts, validated posts, failures and lifecycle |
| `calls.jsonl` | Per-request token accounting, attempts, repairs, latency and failures, including preflight |
| `results.jsonl` | One durable result per question/seed, including partial cancelled trials |
| `summary.json` | Totals with graded denominator, distinct-task count and unknown usage |
| `summary.csv` | One flat row per result for your slide table |
| `figures/cost_and_runtime.png` | Measured tokens and runtime by trial |
| `figures/outcomes.png` | Agreement, correctness, unscored runs and failed turns |
| `demo.html` | Self-contained recorded trace with a task selector and replay |

Check the answer and rationale yourself. Agreement requires explicit ratification
by all three distinct agents on the same active proposal; its author must also
ratify. A revised proposal clears earlier votes. Agreement is not correctness,
and it is not a formal Strong/Ultra-Strong intelligibility result.

References are read only by the evaluator after inference. Automatic grading is
case/whitespace/trailing-punctuation normalized exact matching, not semantic
grading. A correct-looking verbose answer can fail the requested exact format.
Missing references are unscored; `{"manual_review": true}` marks human review
pending. Do not silently change grading rules after looking at model output.
Unknown token counts stay unknown. When a successful request required retries, prior failed attempts have unreported usage; the exact total stays unknown, while known token counts and retry attempts remain available in the records.

The five examples cover arithmetic, simple logic and fault-tolerance reasoning.
They are a curated prototype sample. Describe results as preliminary: “X correct
out of Y graded runs on Z distinct example questions,” with the model and seed.
No recovery comparison or official benchmark accuracy is implemented here.

For a 3–5 minute presentation, use 3–4 slides:

1. **Problem and design:** agents propose, criticize and explicitly endorse answers on a shared board.
2. **Working prototype:** the three roles, real task input, validated posts and separate grading. Show a short replay here.
3. **Measured results:** a small real-run table plus runtime/tokens; discuss one interesting exchange.
4. **Next steps:** historical counterfactual recovery, tool benchmarks and larger controlled experiments.

## 6. If something fails

### Agents keep revising the same answer

A reported bat-and-ball trace showed `prediction.claim` stuck at `0.10` while
the same post's rationale computed `0.05`. This is a claim/explanation mismatch.
The prototype now requests **explanation → prediction → tag → target id** in
both its priority instructions and its optional structured-output schema ordering.
This lets the model write its calculation before choosing the answer/tag. Schema
key ordering is guidance, not a guarantee of semantically consistent model output.
The instructions also say to post a known corrected answer with REVISE, rather
than continue criticizing while restating the disputed answer. A RATIFY must
endorse the existing answer and explanation.

Agent-visible leading claims are also pinned to the controller's active revision
root. The shared in-memory stub can otherwise change its leading claim after a
ratification of an older post; that older endorsement must not redirect the next
agent's view away from the currently tracked proposal. Attempts to endorse a
superseded proposal now trigger a repair request before submission; raw attempts
and repair reasons remain in the trace.

The prototype's `prototype-v6-calculation-chains` prompt policy explicitly resolves
a conflict with the shared PXP guidance: accepting a sound existing proposal uses
RATIFY, including when an agent changes its earlier view. REVISE is reserved for
an actual answer correction or a substantive reasoning correction. Rewording a
sound explanation does not require a new proposal. Votes still clear on every
actual revision, and the runner never converts a revision into an endorsement.

For numerical questions, every role is now asked to define unknowns, formulate
all given constraints, solve them, and substitute proposed values back into all
original constraints. The calculation goes at the start of the rationale so it
survives the board's compact history rendering. This is general guidance; the
runner does not supply the evaluator's answer to the agents. It improves the
prompt but does not guarantee a model will solve every question or agree.
Numerical claims additionally require `CALC: expression = result` evidence.
An exact rational calculator checks each declared equality and the final claim.
Invalid calculations cause a repair request; persistent errors fail the turn.
This verifies arithmetic consistency, not whether the model chose the right
expression for the task. References remain evaluator-only.

If you already uploaded an older source ZIP to Colab:

1. Finish the active run or use the notebook's optional stop cell.
2. Rerun **Upload and verify the current source** with the newly generated
   `outputs/blackboard_prototype_source.zip` from your laptop.
3. Rerun **Install Ollama and configure one model** to refresh `ROOT`-dependent
   environment variables. The already downloaded model is reused.
4. Rerun **Persist results and check inference**, then launch a fresh one-question
   compute run and its viewer. Do not reuse or overwrite the old results.
5. Check the new `manifest.json` contains
   `"prompt_policy": "prototype-v6-calculation-chains"` for ordinary runs, or
   `"prompt_policy": "prototype-recovery-v6-calculation-chains"` for recovery.

If the trace still alternates, inspect the actual claims and rationales. Repeated
correct proposals without ratifications are an agreement failure; repeated
incorrect claims are a reasoning failure. Both remain valid recorded outcomes.
More turns alone do not guarantee a solution. Retain unsuccessful pilot runs and
report prompt/settings changes when comparing their results.

- **No GPU:** select a T4 runtime and rerun setup. Inspect `nvidia-smi`.
- **Backend unreachable:** inspect `/content/blackboard_ollama.log`; confirm the server process is running and `OLLAMA_HOST` agrees with its address.
- **Model absent:** rerun the model pull cell. It must finish before preflight.
- **Model not fully on GPU:** stop other loaded models and GPU consumers, inspect `ollama ps` and `nvidia-smi`, and rerun preflight. Do not claim GPU compute until the placement check passes.
- **Invalid JSON/failed turns:** inspect raw responses and repair events in the trace. These remain failures rather than fabricated contributions.
- **No agreement within nine turns:** inspect the provisional answer and criticism. This is a valid prototype outcome, not proof that compute failed.
- **Output directory exists:** pick a new directory. The notebook does this automatically.
- **Colab iframe unavailable:** inspect `/content/blackboard_demo.log` and the viewer cell. After completion, download the ZIP and open its `demo.html` locally.
- **Disconnect/cancel:** completed results and written trace events remain in Drive. Full mid-trial resume is not implemented; start a fresh run. The notebook's stop cell sends SIGINT to permit clean export.

The setup uses the [official Ollama Linux installer](https://docs.ollama.com/linux)
and Colab's [official iframe serving helper](https://github.com/googlecolab/colabtools/blob/main/google/colab/output/_util.py).
