# Fixes required for the presentation prototype

Prepared: 5 October 2026.

**Status: implemented for the bounded text-only prototype.** See the [working commands and Colab notebook instructions](prototype_quickstart.md). Offline regression checks cover the fixes below; GPU readiness and real-model output must still be checked in Colab before presentation. Broader architecture features remain deferred.

This is the short prototype checklist for the upcoming presentation. The broader [Student 3 readiness document](student3_compute_readiness.md) remains the roadmap for historical replay and research experiments.

Following the later request for a counterfactual demonstration, a separate
[natural recovery runner](prototype_recovery_demo.md) is also available. It
supports bounded historical replay and live confirmation after an observed
conflict. Full baseline comparisons, causal credit and the distributed recovery
daemon remain part of the broader roadmap.

## 1. The prototype we are aiming to deliver

The first working prototype will:

1. Accept an actual text reasoning question and its agent-visible context.
2. Run three stable persona agents using the shared Qwen model.
3. Show validated prediction/explanation exchanges on an in-memory blackboard.
4. Record the final proposed answer and whether all three agents explicitly agreed on it.
5. Grade supported answers separately from agreement.
6. Save responses, failures, usage, and results as the run progresses.
7. Produce a small results table, charts, and a saved trace suitable for a presentation demo.

**The first prototype will demonstrate real-model deliberation. Historical counterfactual recovery will remain a separately labelled unfinished feature.** Enabling a capability flag alone does not constitute a recovery implementation or an ablation experiment.

The initial compute target is three distinct tasks with one seed each. More tasks or a second seed will be added only after measuring runtime. Repeated seeds will be reported as additional runs, not additional distinct questions.

## 2. Implementation approach

Build a small `scripts.prototype_run` entrypoint using the already working Student 2 components:

- `PEXAgent` and its bounded parser/semantic repair loop.
- `InferenceSettings`, preflight, the shared model broker, and its usage ledger.
- `agents.board_client.InMemoryBoard` for sequential, zero-based commits.
- The existing persona and agent interfaces, with an explicit reasoner/critic/verifier roster.

Use a fixed round-robin order for ordinary turns. Do not depend on the incomplete recovery daemon, tool environments, Redis deployment, or live dashboard connection for this presentation path.

This limits the repair surface while retaining real model calls, actual questions, agent roles, blackboard history, and measured instrumentation. Existing smoke commands will remain available.

The existing batch runner must not be used for empirical presentation results until its task delivery, accounting, and outcome semantics are repaired. The prototype entrypoint will provide the supported compute path for the deadline.

## 3. Essential fixes before compute

### Fix 1 — Deliver actual questions and preserve task identity

**Current problem:** the live batch executor prompts agents with `Benchmark Problem: <task_id>` rather than the actual question. Its CLI expands a few sample questions into renamed scenarios, which does not create new benchmark tasks.

**Change:** accept a JSON list of tasks with `task_id`, `question`, and optional object-valued `context`. Validate non-empty questions and unique task IDs. Put only the question and permitted context on the board. Preserve task IDs unchanged across seed repetitions.

Reference answers will come from a separate evaluation JSON file keyed by task ID. That file will never be passed to an agent, prompt, or board snapshot. Reject known reference-answer fields in agent-visible task context instead of silently exposing them.

**Acceptance check:** inspect captured requests and confirm the actual question is present and evaluation answers are absent. Changing only the evaluation file must not change inference inputs.

### Fix 2 — Provide a runnable, bounded trial loop

**Current problem:** importing the existing batch runner and using a supplied executor can raise `NameError` because `asyncio` is only imported in its command-line block. Its live loop also silently continues after some failures and misclassifies an exhausted turn budget as deadlock.

**Change:** add the missing module-level import as a small compatibility fix. Use the dedicated prototype loop for the presentation: three agents, fixed turn order, explicit offline/live execution, and no automatic backend fallback. Start with a nine-turn limit per trial and a shared 100-backend-attempt / 20-minute ceiling. Check the budget before every backend attempt, including retries and repair calls; cap request waiting by the remaining deadline.

Record distinct outcomes: agreement, turn/call/time limit reached, inference failure, observed disagreement stall, and cancellation. Invalid model output will remain a recorded failed turn; repeated backend unavailability will terminate the trial rather than generate mock results.

**Acceptance check:** run the entrypoint both as a command and through an imported function. Force invalid responses and an exhausted budget; confirm recorded failures and explicit termination rather than fabricated success.

### Fix 3 — Make agreement reporting conservative

**Current problem:** the scheduler can accept trailing ratifications of different claims as consensus. The live executor can declare recovery immediately after a `REVISE`. The scheduler smoke path also produced duplicate turn indices.

**Change:** use the direct in-memory board's sequential zero-based indices. Track the current proposal using its contribution ID. A revision creates a new proposal and clears votes. An agent's latest validated ratification must resolve to that current proposal and match its claim. A criticism withdraws that agent's vote. Require explicit ratifications from all three distinct roster agents before reporting three-agent answer agreement; proposing an answer does not count as ratifying it.

Record agreement separately from correctness. Do not automatically assign Strong or Ultra-Strong intelligibility labels. Formal explanation-level PXP consensus and its shared contract remain part of the later architecture work.

Follow-up: the prototype prompt now explicitly overrides the shared instruction
to revise when persuaded. Accepting a sound existing proposal uses RATIFY; a new
answer or substantive reasoning correction uses REVISE. Numerical verification
requires solving and substituting into all original constraints. The manifest
records this policy as `prototype-v7-fraction-grouping`. Numerical posts now
require exact-calculator-checked `CALC` evidence; false equalities and stale
ratifications trigger repair before commitment. The runner still cannot
guarantee model correctness and does not supply reference answers to agents.

**Acceptance check:** two voters cannot complete agreement; votes on different proposals cannot combine; a revised proposal cannot inherit old votes; an incorrect unanimous answer remains incorrect when graded.

### Fix 4 — Grade final answers without false positives

**Current problem:** the existing evaluator accepts an empty prediction as a normalized match. Ingestion can discard `0` and `False` answers. The current live trial result does not preserve a final answer for independent grading.

**Change:** save the active proposal's claim as `final_answer`; mark it provisional when no three-agent agreement was reached. Use `null` when no valid proposal exists. Fix the shared evaluator's empty-string match and falsey-answer handling, and preserve `0` and `False` in the affected ingestion adapters.

For the prototype's automatic grading, use simple tasks with explicitly defined answer strings or choice identifiers. Compare exact answers after case, surrounding whitespace, and trailing-punctuation normalization; never use arbitrary substring matching. Responses requiring semantic interpretation will be marked for manual review, with the reviewed decision recorded separately. Missing references will be unscored.

**Acceptance check:** an empty answer fails against a non-empty reference; `0` and `False` survive parsing; a partial answer does not pass because it is a substring; agreement and correctness can differ.

### Fix 5 — Report actual per-trial usage and runtime

**Current problem:** the live batch executor reads cumulative broker usage. Identical offline trials reported 12,188 and then 24,376 tokens even though their individual workloads were equivalent.

**Change:** wrap each trial in its own `usage_scope` and obtain totals from `broker.trial_usage(trial_id)`. Record input/output tokens, backend attempts, repairs, failures, and elapsed time. Keep unknown usage as `null`; never turn it into zero. Report preflight/model-loading time separately from task runtime.

The prototype result will use a separate versioned result model so it does not need to invent an intelligibility assessment or force unknown token totals into the legacy `TrialResult` integer field.

**Acceptance check:** identical scripted trials report equal individual usage; per-trial totals reconcile with their call records; backend attempts include retries; missing token counts remain unknown.

### Fix 6 — Save useful work before the batch finishes

**Current problem:** the old batch exports results after all trials finish. Smoke-run raw responses, contributions, failures, and summaries are mostly written at completion. Reusing an output directory can mix appended ledger records with overwritten summaries.

**Change:** create a fresh run directory and refuse accidental reuse. Write a run manifest first, append raw responses and validated contributions/failures as they occur, and flush a result record after each completed trial. Keep partially completed traces on exceptions or cancellation. Identify trials by task ID and seed within the run.

Save beneath `outputs/presentation/<run-id>/`, or an explicitly selected persistent notebook destination. Restart interrupted trials from a clean board; full mid-trial resume is outside this prototype.

**Acceptance check:** interrupt a two-task run during its second task. The first result and second task's already-written trace must remain readable. Reusing an existing run directory must fail clearly.

### Fix 7 — Run the intended branch on the GPU

**Current problem:** the notebook defaults to `main`, which differs from the currently audited `s3-compute-readiness` branch. Editing local files does not automatically update a cloud notebook's checkout.

**Change:** the presentation notebook uploads a source ZIP of the current workspace, verifies its file hashes, and records its source digest and base commit. This includes uncommitted fixes and avoids cloning stale `main`. The older Student 2 notebook names `s3-compute-readiness`; it requires the intended code to be available remotely. The run manifest also records model digest, effective settings, task IDs, and seeds.

Run preflight on the compute machine and inspect its GPU warnings as well as required checks. A backend being reachable does not itself prove GPU execution. Use a unique persistent output destination and explicitly copy or download the artifacts.

**Acceptance check:** the notebook contains the new runner, its recorded revision matches the intended implementation, one real-model task finishes, and its artifacts can be opened outside that runtime.

### Fix 8 — Create presentation outputs from the actual run

**Current problem:** dashboard ablation metrics are hardcoded sample data, its Run button only logs a message, and stored example plots are simulation artifacts.

**Change:** generate a CSV summary and standalone charts directly from prototype results. Include distinct-task count, run count, graded correctness with its denominator, three-agent answer agreement, tokens, runtime, repairs, and failures. Report unscored tasks and unknown usage explicitly.

Create a readable HTML trace viewer for a completed trial showing the question, chronological agent exchanges, final answer, agreement, grading result, and measured usage. It will open as a local artifact and serve as the optional demo without depending on a GPU or WebSocket server during the presentation. Any offline scripted trace will be visibly labelled as scripted.

**Acceptance check:** every displayed number reconciles with the saved results. The viewer opens locally, and the selected example's outcome matches its trace. No simulated ablation percentages appear as empirical findings.

## 4. Proposed inputs, command, and artifacts

The interface below is implemented through `scripts.prototype_run`.

```bash
python -m scripts.prototype_run \
  --backend ollama \
  --tasks tasks.json \
  --references references.json \
  --seeds 42 \
  --max-turns 9 \
  --out outputs/presentation/run-001
```

`tasks.json` contains a list of agent-visible task objects. `references.json` maps their IDs to evaluator-only reference answers. References are optional; omitted references produce unscored results.

Expected artifacts:

| Artifact | Purpose |
|---|---|
| `manifest.json` | Execution mode, revision, model/settings, task identities, seeds, and limits |
| `calls.jsonl` | Actual per-call usage, attempts, latency, and failures |
| `trace.jsonl` | Incremental raw responses, repairs, contributions, and lifecycle records |
| `results.jsonl` | Incremental completed-trial outcomes and final answers |
| `summary.csv` | Presentation results with denominators and missing values preserved |
| `figures/` | Charts generated from this run |
| `demo.html` | Locally viewable representative trace with no inference dependency |

## 5. Work deliberately deferred

These do not block the text-only presentation prototype:

- Full historical baseline/intervention replay, candidate selection, measured attribution, and live recovery confirmation.
- Redis atomicity, distributed concurrency, and canonical parity across every store.
- Official KramaBench/MedAgentBench tool execution and environment checkpointing.
- Full dashboard compute controls, cross-process streaming, and reconnect repair.
- A four-condition capability ablation, official benchmark-performance claims, and statistical inference from a large study.

The prototype will keep recovery disabled and report that fact. It will not count ordinary additional discussion as counterfactual recovery. A later scripted recovery demonstration must be labelled separately from the real-model result set.

## 6. Go/no-go checklist and time limit

Compute starts with one real-model task only after:

- [x] Offline regression checks pass for task delivery, agreement, grading, accounting, limits, and persistence.
- [x] Existing tests remain green except any explicitly corrected obsolete expectations.
- [ ] The intended implementation is available in the compute environment.
- [ ] Preflight succeeds and actual GPU/model placement is inspected.
- [ ] One complete live trial produces readable traces and a reconciled result.

Then run three distinct tasks first and expand only if measured runtime permits.

**Implementation is complete for this bounded prototype; local regressions pass.** GPU acceptance remains a compute-machine check. Start with one question in the provided Colab notebook, inspect the real result, then run the small batch. If live inference cannot be completed in time, distinguish the scripted pipeline check from real model findings.

Reserve the final hour for slides and rehearsal. A complete trace and a small truthful results table are the minimum presentation deliverables; a large unfinished batch is not the goal.
