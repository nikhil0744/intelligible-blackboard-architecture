# KramaBench runs in Colab

Use [colab_benchmarks.ipynb](../../notebooks/colab_benchmarks.ipynb) to run persona agents on selected official KramaBench tasks. The notebook prepares pinned data, starts local Qwen inference, saves attempts to Drive, and displays the recorded discussion. It runs entirely inside the Colab runtime.

**Validation:** the original pilot has run on a managed Colab GPU. The expanded 20-task profile requires a new end-to-end GPU run. Save the run manifest alongside the results to record the exact configuration and task selection.

## Scope

| Feature | Current behavior |
| --- | --- |
| Default notebook profile | `kramabench20`, a fixed selection of 20 official CSV tasks with numeric, text and list answers |
| Original legal pilot | `kramabench`: `legal-easy-4`, `legal-easy-5`, `legal-easy-21` |
| Inference | One local `qwen2.5:7b-instruct-q4_K_M` shared sequentially |
| Agents | Data analyst, data engineer/critic, general reasoner |
| Execution | Actual `PEXAgent` validation/inference, in-memory board, deterministic scheduler, controlled tools |
| Grading | Pinned upstream Success, RAE score and F1, depending on answer type. The reported correct count means full credit. |
| Recovery | Bounded evidence backtracking enabled by default; use the notebook switch or CLI option `--no-recovery` for baseline runs |
| MedAgentBench | Unavailable pending its official environment/evaluator inside Colab |
| Presentation | Saved trace table and recorded playback through the existing HTML viewer |

These are profile results, not the official aggregate. An incorrect answer remains incorrect even if all agents agree. The [resource manifest](../../src/benchmarks/resources.json) pins upstream commits, input/evaluator sizes and SHA-256 hashes, the Ollama archive, and Qwen digest. It contains no solutions. Setup rejects partial/incompatible caches and does not silently substitute another model.

## Run in Colab

1. Open the notebook and choose **Runtime > Change runtime type > GPU**.
2. Run bootstrap. It checks out the notebook's immutable project revision and installs dependencies.
3. Mount Drive. Cell 2 defaults to `BENCHMARK = kramabench20` and `TASK_COUNT = 20`. Select 1 first for a quick GPU check, or select 2, 3, 5 or 10 for a shorter batch. For the original pilot, choose `BENCHMARK = kramabench` and at most 3 tasks. To select specific tasks, enter their IDs in `TASK_IDS`, separated by commas. An explicit list takes precedence over the count.
4. Run preparation in cell 3 and check the selected profile and task IDs. Run cell 4 to start the tasks one at a time. Larger batches print the task position and saved outcome. Set `VERBOSE_BATCH = True` to see individual tool calls and posts. Every attempt keeps its full event log.
5. Review each submitted answer, grading metric and score in cell 5. Set `SHOW_LATEST_TRACE = True` to open the latest trace in a larger batch. In cell 6, choose a completed task or retry from the playback dropdown. Download the bundle and open `presentation/index.html` from the extracted ZIP to switch tasks offline.

Setup downloads about 1.5 GB of compressed Ollama runtime and 4.7 GB of model weights. Model files stay on local runtime disk; Drive holds the small benchmark cache and results. A new runtime downloads the model again. Setup checks GPU, RAM and disk, then uses the official user-space Linux archive without systemd or Docker. This recipe still requires fresh-Colab validation.

The per-task defaults are 15 discussion turns, 24 tool rounds, 90 backend attempts and 20 minutes. Cell 2 sets a 180-minute deadline for each notebook invocation by default. The equivalent CLI default is 60 minutes. Each task uses at most the invocation's remaining time. If the runtime disconnects or the batch reaches its deadline, resume the saved run to continue the pending tasks. Actual batch time depends on the GPU and the number of model and tool calls.

Tool timeouts terminate and join the worker. A client inference timeout cannot guarantee Ollama stops an already submitted request; unreported usage remains unknown. The single-request server setting serializes inference.

## Commands and resume

Install with Python 3.11+ using normal project setup. Local runs require an already running Ollama and explicitly pulled model; the Colab notebook handles those setup actions.

```bash
python -m benchmarks.run prepare --cache-dir outputs/benchmark-cache
python -m benchmarks.run doctor
python -m benchmarks.run run --limit 1 --cache-dir outputs/benchmark-cache --results-dir outputs/benchmark-results
python -m benchmarks.run run --limit 3 --cache-dir outputs/benchmark-cache --results-dir outputs/benchmark-results
python -m benchmarks.run prepare --benchmark kramabench20 --cache-dir outputs/benchmark-cache
python -m benchmarks.run run --benchmark kramabench20 --limit 1 --cache-dir outputs/benchmark-cache --results-dir outputs/benchmark-results
python -m benchmarks.run run --benchmark kramabench20 --run-timeout-s 10800 --cache-dir outputs/benchmark-cache --results-dir outputs/benchmark-results
python -m benchmarks.run resume --run-dir PATH --cache-dir outputs/benchmark-cache
python -m benchmarks.run summarize --run-dir PATH
```

[Inference settings](inference.md) use `LLM_*` variables. The real command permits local Ollama only, one call at a time. Missing model/GPU/evaluator/data stops execution; there is no mock, paid API, simulation or model fallback. `--cpu-smoke` explicitly permits a CPU machinery check, marked in the manifest. `--temporary` explicitly permits ephemeral Colab storage; download it before disconnecting.

`--task-ids legal-easy-4 legal-easy-21` selects explicit tasks when present in the chosen profile. `--limit N` selects the first `N` tasks in the manifest. Omitting it selects the full profile. The ordered list is saved before solving.

Resume uses the saved profile, selection, seed, budgets, personas and inference settings. In Colab, set `MODE = resume` and put the full run folder in `RESUME_RUN_DIR`. The profile and count entered for new runs do not change this selection. Keep the recorded project revision and CPU/GPU mode. Resume verifies Python/package versions, data/evaluator resources, model digest, and effective settings.

Committed attempts, including incorrect results, are skipped. Partial/interrupted tasks restart with new attempt IDs and clean inputs; old records remain. Committed model/output/time/tool failures are skipped by default. `resume --retry-failures` explicitly retries those failures and retains both attempts; it does not retry a normally completed incorrect answer. Aggregates use the latest explicit attempt per selected task; usage includes all attempts.

Use one writer per saved run in one Colab runtime at a time. A local advisory lock rejects duplicate writers within that runtime; it is not a distributed lock between VMs.

## Task execution and grading

Agents receive only the official question, permitted filenames and answer type. Solutions, answer-bearing subtasks and grading code stay in the private preparation cache. Every task gets fresh copies of only its allowed CSV inputs.

The scheduler rotates three fixed personas. Agents return typed tool requests, PXP contributions or final submissions of the required answer type. The controller logs each request's board version/turn and stable ID. Re-delivering that ID returns its saved observation. Intentional repeated computations receive new IDs and consume budget.

Tools support listing, CSV inspection, filters, grouping, aggregation and bounded arithmetic, including header preambles and Windows-1252 inputs. No generated code, shell, expression, network request or arbitrary path is accepted. Trusted operations run in a child with CPU/memory limits. This is a restricted interface, not a general Python sandbox. Tools cannot read evaluator files or write Drive results.

Full observations are saved; recent bounded observations return to prompts. Excerpts are marked, with a hint to narrow the query. Prompt truncation is recorded in contribution metadata.

The opening agent requests tools until a CSV query returns usable data. Scope checks compare recognizable question qualifiers with the selected rows and flag mismatches in the trace. The analyst can refine its query before submitting an answer for panel review. The tool budget allows up to eight distinct calls per turn. The controller records the query's request ID at the handoff. A listing, inspection or failed query cannot open discussion. If evidence collection exhausts its tools, the run records `insufficient_evidence`. Two repeated rounds of the same disagreement without new successful queries stop as `stalled`. These conditions can trigger a bounded recovery investigation when recovery is enabled.

A revision starts a fresh claim. All three agents, including its author, must explicitly ratify that claim after the latest revision, and a final submission is required. Disagreement clears prior votes. Incompatible claims cannot establish consensus. On a deadlock, a repeated disagreement, or an investigation that exhausts its turn without usable evidence, the enabled controller may start one new branch before the opening claim. It carries forward bounded observations and challenges and requires a different in-scope CSV query. A candidate replaces the original submission only after fresh votes from all three agents. The branch shares the original call, time, turn and tool budgets. A malformed model response remains an output error. For a paired comparison create two new runs with the same task IDs, seed, model and budgets. Toggle only recovery. Record attempts, accepted answers, consensus and independent grades from both runs.

The final output is frozen before grading. The grader loads the checksum-pinned local metric definitions and dispatches by answer type. The results include the primary metric name and score, with full-credit outcomes counted as correct. Approximate and list metrics can award partial credit. Grading keeps the reference answers separate from model prompts. Hosted judges, pipeline grading and official aggregate scores are outside these profiles.

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
| Recorded HTML viewer | Standalone timeline over verified recorded events; the legacy viewer remains available to existing callers |

## Acceptance evidence to collect

- Fresh managed GPU runtime: prepare and run one task; inspect Qwen digest/GPU use, independent grade and verified files in Drive.
- Run a short multi-task selection with `STOP_AFTER_SAVED = 1`, reconnect in a new runtime, set `STOP_AFTER_SAVED = 0`, and resume the same folder. Confirm committed tasks are skipped and previous attempts remain.
- Run the fixed 20-task selection. If you need more than one invocation, keep using the same saved folder until all 20 tasks have a recorded outcome.
- Rebuild summary/CSV/optional figure; compare a recorded viewer trace with `events.jsonl`.
- Retain manifest and setup/service errors. A valid incorrect grade still tests the machinery.

Offline CI uses explicitly scripted fixtures, without downloads/GPU. It verifies integration, answer isolation, tool access/timeouts, consensus, accounting, partial caches, interruption/resume, configuration mismatch and record corruption. It does not replace Colab acceptance.
