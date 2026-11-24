# Colab benchmark pilot

Use [colab_benchmarks.ipynb](../../notebooks/colab_benchmarks.ipynb) to run actual persona agents against a small, pinned KramaBench profile. It prepares data, starts local inference, saves attempts to Drive, and displays the recorded discussion. No provider API key, Redis, React server, or external inference server is required.

**Validation:** offline integration and local official-asset preparation/grading checks pass. Fresh managed Colab GPU execution, Drive persistence across reconnects, and model quality remain unverified. This pilot does not complete the two-benchmark study.

## Scope

| Feature | Current behavior |
| --- | --- |
| KramaBench | `legal-csv-numeric-v1`: `legal-easy-4`, `legal-easy-5`, `legal-easy-21` |
| Coverage | Three of 30 legal-domain top-level tasks; other domains/metrics excluded |
| Inference | One local `qwen2.5:7b-instruct-q4_K_M` shared sequentially |
| Agents | Data analyst, data engineer/critic, general reasoner |
| Execution | Actual `PEXAgent` validation/inference, in-memory board, deterministic scheduler, controlled tools |
| Grading | Pinned upstream `Success` definitions, numeric answers; no hosted judge/pipeline grading |
| Recovery | Disabled; zero capable agents and zero recovery calls |
| MedAgentBench | Unavailable pending its official environment/evaluator inside Colab |
| Presentation | Saved trace table and recorded playback through the existing HTML viewer |

These are profile results, not the official aggregate. An incorrect answer remains incorrect even if all agents agree. The [resource manifest](../../src/benchmarks/resources.json) pins upstream commits, input/evaluator sizes and SHA-256 hashes, the Ollama archive, and Qwen digest. It contains no solutions. Setup rejects partial/incompatible caches and does not silently substitute another model.

## Run in Colab

1. Open the notebook and choose **Runtime > Change runtime type > GPU**.
2. Run bootstrap. It checks out the notebook's immutable project revision and installs dependencies.
3. Mount Drive, select one task for the first run, and run preparation.
4. Run the experiment. Claims, PXP tags, requests and observations print as records are saved.
5. Review grades and a saved trace; optionally download the bundle.

Setup downloads about 1.5 GB of compressed Ollama runtime and 4.7 GB of model weights. Model files stay on local runtime disk; Drive holds the small benchmark cache and results. A new runtime downloads the model again. Setup checks GPU, RAM and disk, then uses the official user-space Linux archive without systemd or Docker. This recipe still requires fresh-Colab validation.

Defaults: 15 discussion turns, 24 tool rounds, 90 backend attempts, 20 minutes per task, and 60 minutes per batch invocation. These are this project's pilot limits, not official execution limits. Each task is capped at the invocation's remaining time. A runtime estimate needs real measurements first.

Tool timeouts terminate and join the worker. A client inference timeout cannot guarantee Ollama stops an already submitted request; unreported usage remains unknown. The single-request server setting serializes inference.

## Commands and resume

Install with Python 3.11+ using normal project setup. Local runs require an already running Ollama and explicitly pulled model; the Colab notebook handles those setup actions.

```bash
python -m benchmarks.run prepare --cache-dir outputs/benchmark-cache
python -m benchmarks.run doctor
python -m benchmarks.run run --limit 1 --cache-dir outputs/benchmark-cache --results-dir outputs/benchmark-results
python -m benchmarks.run run --limit 3 --cache-dir outputs/benchmark-cache --results-dir outputs/benchmark-results
python -m benchmarks.run resume --run-dir PATH --cache-dir outputs/benchmark-cache
python -m benchmarks.run summarize --run-dir PATH
```

[Inference settings](inference.md) use `LLM_*` variables. The real command permits local Ollama only, one call at a time. Missing model/GPU/evaluator/data stops execution; there is no mock, paid API, simulation or model fallback. `--cpu-smoke` explicitly permits a CPU machinery check, marked in the manifest. `--temporary` explicitly permits ephemeral Colab storage; download it before disconnecting.

`--task-ids legal-easy-4 legal-easy-21` selects explicit tasks. Otherwise `--limit` must be 1–3; ordering follows the manifest and is saved before solving. No resampling or hidden task replacement occurs.

Resume uses the saved selection, seed, budgets, personas and inference settings. Changed source, Python/package versions, data/evaluator resources, model digest or effective settings cause rejection. Keep the recorded project revision; do not pull changing code into an active experiment. Use a compatible environment or start a new run when versions change.

Committed attempts, including incorrect results, are skipped. Partial/interrupted tasks restart with new attempt IDs and clean inputs; old records remain. Committed model/output/time/tool failures are skipped by default. `resume --retry-failures` explicitly retries those failures and retains both attempts; it does not retry a normally completed incorrect answer. Aggregates use the latest explicit attempt per selected task; usage includes all attempts.

Use one writer per saved run in one Colab runtime at a time. A local advisory lock rejects duplicate writers within that runtime; it is not a distributed lock between VMs.

## Task execution and grading

Agents receive only the official question, permitted filenames and answer type. Solutions, answer-bearing subtasks and grading code stay in the private preparation cache. Every task gets fresh copies of only its allowed CSV inputs.

The scheduler rotates three fixed personas. Agents return typed tool requests, PXP contributions or numeric final submissions with PXP contributions. The controller logs each request's board version/turn and stable ID. Re-delivering that ID returns its saved observation; intentional repeated computations receive new IDs and consume budget.

Tools support listing, CSV inspection, filters, grouping, aggregation and bounded arithmetic, including header preambles and Windows-1252 inputs. No generated code, shell, expression, network request or arbitrary path is accepted. Trusted operations run in a child with CPU/memory limits. This is a restricted interface, not a general Python sandbox. Tools cannot read evaluator files or write Drive results.

Full observations are saved; recent bounded observations return to prompts. Excerpts are marked, with a hint to narrow the query. Prompt truncation is recorded in contribution metadata.

The opening agent requests tools until a CSV query returns matching data that passes the available scope checks. The checks compare recognizable question qualifiers with the selected rows and flag mismatches in the trace. The analyst can refine its query before submitting a numeric answer for panel review. The tool budget allows up to eight distinct calls per turn. The controller records the query's request ID at the handoff. A listing, inspection, failed query, or empty query cannot open discussion. If evidence collection exhausts its tools, the run records `insufficient_evidence`. Two repeated rounds of the same disagreement without new successful queries stop as `stalled`. Both statuses are distinct from deadlock recovery and answer correctness.

A revision starts a fresh claim. All three agents, including its author, must explicitly ratify that claim after the latest revision, and a final submission is required. Disagreement clears prior votes. Incompatible claims cannot establish consensus. The baseline stops at consensus, unresolved disagreement, resource limits, error or interruption. Recovery comparisons stay unavailable until baseline Colab/resume acceptance and separate recovery verification.

The final numeric output is frozen before grading. The grader extracts only checksum-pinned `str_to_float`, `try_convert_to_number` and `Success` definitions, avoiding upstream imports that load hosted-model grading paths. Grading never feeds labels to agents and follows the pinned metric's numeric coercion/tolerances. Other official metrics are excluded.

No final submission is reported as such with score zero; it is not counted as a graded answer. Broken grading is an evaluation error. Consensus, correctness, recovery and intelligibility are separate; formal intelligibility is unassessed.

## Files and accounting

```text
blackboard_benchmarks/
  cache/kramabench/<revision>/<profile>/
  runs/<run-id>/
    manifest.json
    selection.json
    attempts/<attempt-id>/
      attempt.json
      events.jsonl
      calls.jsonl
      final_output.json
      result.json
      complete.json
    summary.json
    results.csv
    presentation/<attempt-id>/viewer.html
    presentation/<attempt-id>/trace.json
    figures/outcomes.png
```

Events contain actual model prompts/responses, repairs, observations and contributions. Completion markers checksum authoritative files. Corruption stops reporting/resume instead of dropping a task. Outputs/results use temporary replacements, flushing and read-back verification. CSV/summaries/figures can be rebuilt from saved records without inference or data access. Generated files, datasets, model weights and internal plans stay out of Git.

Events flush immediately; call records append after each broker call and flush before attempt commit. Abrupt termination may lose the in-flight call and unacknowledged Drive writes. Interrupted records are retained and exact all-attempt tokens remain unknown. Checks detect missing/inconsistent files; they cannot guarantee remote replication of every acknowledged Drive write.

Per-attempt usage includes parse repairs and backend retries. Unknown counts stay `null`; known-only subtotals are labelled. No token estimates or dollar costs are fabricated. The local grader makes no model calls. The readiness probe is setup overhead, recorded separately from task costs.

Reports retain all selected tasks and show correct / all selected and correct / scored. Pending, interrupted and ungraded tasks remain in the first denominator. Explicit retries preserve earlier cost/failures. Recorded playback is an artifact viewer, not live browser task launching or the authoritative store; demo recovery scenarios are removed from its playback.

## MedAgentBench and preserved work

The pinned upstream [setup](https://github.com/stanfordmlgroup/MedAgentBench/tree/99260117137b09f04837a8c18d18a1107efa55ae) requires a Docker-hosted FHIR service plus separately distributed `refsol.py`. Configuration points at `localhost:8080/fhir/` and official task/function files. Its POST tool validates/simulates acceptance rather than necessarily mutating records; a future adapter must preserve this behavior.

Source, configuration and evaluator dispatch were inspected. That setup does not provide a verified native-process recipe for the same patient assets, and its Docker deployment/evaluator have not been demonstrated in managed Colab here. No substitute server or reference grader was introduced. MedAgentBench commands explain this blocker and exit. Patient lookup, action semantics, reference grading, and combined resource use need a real feasibility run before enabling it.

| Component | Reuse decision |
| --- | --- |
| S2/S3 agents, validation, prompt budgeting, settings, accounting (`79345bb`) | Adapted/tested against the canonical board/scheduler |
| S2/S3 panel roles/capabilities | Reused; capability remains separate from role |
| Archived prototype runner/recovery | Excluded; not evidence of paired historical intervention/replay |
| Prior-main environment/evaluator wrappers | Inspected; simulations are not official Colab deployments |
| Existing ingesters | Preserved for older formats; pinned adapter allowlists official query fields/private labels |
| Existing mscore batch/exporter | Preserved as historical demo; unused by real benchmark execution/export |
| Existing HTML viewer | Reused with an adapter over verified recorded events |

## Acceptance evidence to collect

- Fresh managed GPU runtime: prepare and run one task; inspect Qwen digest/GPU use, independent grade and verified files in Drive.
- Run three tasks, stop after `attempt_saved`, reconnect in a new runtime, and resume the same selection. Confirm committed tasks are skipped and previous attempts remain.
- Rebuild summary/CSV/optional figure; compare a recorded viewer trace with `events.jsonl`.
- Retain manifest and setup/service errors. A valid incorrect grade still tests the machinery.

Offline CI uses explicitly scripted fixtures, without downloads/GPU. It verifies integration, answer isolation, tool access/timeouts, consensus, accounting, partial caches, interruption/resume, configuration mismatch and record corruption. It does not replace Colab acceptance.
