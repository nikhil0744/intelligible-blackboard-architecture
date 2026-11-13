# Project status and implementation review

**Review date:** 12 November 2026  
**Repository:** `nikhil0744/intelligible-blackboard-architecture`  
**Local branch:** `prototype-live-demo`  
**Reviewed implementation commit:** `5c0036d5694504d1f4e1dea6be6d98d08034abcf`  
**Scope:** current local implementation, tests, configuration, documentation, and available example artifacts. Existing documentation changes are additional to the commit above. Remote branches, GitHub Issues, and deployed services were not inspected.

This document is a review of current behavior and improvement opportunities. It is not a new implementation plan, a student assignment list, or a claim that the recommended changes have been made. The old plans are [archived and inactive](archive/README.md).

## 1. Overall assessment

The project is a substantial modular research prototype. It contains real implementations of shared contracts, model calling, persona prompts, agent turns, storage, scheduling, deadlock diagnosis, sandbox synthesis, scoring, telemetry endpoints, visualization, dataset parsing, and analysis utilities.

The strongest evidence is for component behavior under controlled offline inputs. The main weakness is agreement between components: the available agent, board, scheduler, recovery, benchmark, and visualization paths do not yet form one verified end-to-end execution system.

**Current maturity: useful component prototype; incomplete integration; experimental results are not yet evidence of the proposed system's effectiveness.**

This does not justify discarding the codebase. Much of the implementation is reusable. Improving shared interfaces, validation, and measurement would address the most consequential gaps before expanding model variety or running larger batches.

### What can currently be claimed

- There is a functioning offline test suite covering many individual components and selected mock-based integrations.
- Persona agents can generate and submit structured contributions through their dedicated development board with a mock backend.
- The scheduler and recovery daemon have controlled fixture/mock integration coverage.
- The telemetry service can accept explicit events and deliver them over its API/WebSocket interfaces in the tested process arrangement.
- The analytics and export utilities can process the repository's simulation records.

### What has not been established

- A shared real-agent execution path that works consistently across the supported stores and scheduler.
- Faithful historical counterfactual intervention with a matching unchanged baseline continuation.
- Independently verified live recovery following application of a sandbox proposal.
- Official benchmark execution, tool interaction, and scoring across all three named datasets.
- Accurate cross-process telemetry feeding the dashboard from the reasoning engine.
- Measured improvements in correctness, recovery efficiency, knowledge transfer, or formal intelligibility.

## 2. Verification performed and its limits

### Fresh checks for this review

| Check | Result | What it establishes |
| --- | --- | --- |
| Python test suite | **179 passed, 1 warning, 4.95 seconds** | Existing assertions pass in the current local environment. |
| Contract schema comparison | **14 of 14 schemas match** | Committed JSON schema exports agree with current Pydantic models. |
| Additional offline probes | Multiple failures reproduced; see section 5 | Important cases are missing from, or not rejected by, the current suite. |
| Runtime availability | Python **3.14.10**; Node/npm unavailable on PATH | Backend tests could run; frontend execution could not be verified here. |
| Example record inspection | 500 preserved records; 8 local output records | Both inspected JSON collections identify their results as simulated. |

The test command was:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/blackboard-review-mpl \
timeout 30s .venv/bin/python -m pytest -q -p no:cacheprovider
```

It ran outside the sandbox because streaming `TestClient` checks hang inside the restricted environment. The successful outside-sandbox run means that timeout is not counted as an implementation defect. Bytecode and pytest cache writes were disabled; generated test artifacts use temporary locations.

The warning is a FastAPI/Starlette test-client deprecation concerning `httpx` and recommending `httpx2`. It did not cause a test failure. Dependency compatibility merits review separately; no dependency changes were made.

### Boundaries of the evidence

- No real Ollama, LiteLLM provider, GPU, or hosted-model run was performed for this review.
- Redis behavior was probed through `InMemoryRedisClient`; real Redis concurrency was not tested.
- No frontend build, browser interaction, or production deployment was executed.
- Python 3.11/3.12 appear in CI configuration, but this local Python 3.14 run does not establish those jobs' results.
- Research references and dataset descriptions were inspected as local documentation, not independently checked against current external publications or releases.
- A passing test proves only its asserted behavior. Some tests explicitly accept behavior that this review identifies as unsuitable for valid experiments.

## 3. Architecture as it exists

### Intended information flow

```text
Dataset item
    -> Normalized task
    -> Shared board + scheduler + persona agents
    -> Disagreement detection
    -> Isolated recovery exploration
    -> Revision application and fresh live agreement
    -> Independent outcome evaluation
    -> Recorded telemetry, dashboard, and analytics
```

The repository has implementations for most boxes in this diagram. Several arrows are missing, incompatible, or replaced by shortcuts in the runnable demonstrations.

### Component inventory

| Area | Main location | Current assessment |
| --- | --- | --- |
| Shared data contracts | [`src/contracts/`](../src/contracts/) | Substantial typed models and synchronized exports; relational invariants need stronger enforcement. |
| Blackboard/storage | [`src/blackboard/store.py`](../src/blackboard/store.py), [`src/storage/redis_store.py`](../src/storage/redis_store.py), [`src/agents/board_client.py`](../src/agents/board_client.py) | Three overlapping board paths with different type, indexing, and validation behavior. |
| Scheduling | [`src/scheduler/`](../src/scheduler/) | Queue, policies, terminal heuristics, and recovery hooks exist; real-agent compatibility and consensus semantics are incomplete. |
| Inference | [`src/llm_broker/`](../src/llm_broker/) | Useful backend abstraction, bounded concurrency, retries, and usage tracking. Actual provider behavior is unverified here. |
| Agents/prompts | [`src/agents/`](../src/agents/), [`src/prompts/`](../src/prompts/) | Structured lifecycle, personas, parser, and repair loop work in the dedicated development path. Semantic repairs and capability roles create problems. |
| Recovery | [`src/counterfactual/`](../src/counterfactual/), [`src/sandbox/`](../src/sandbox/) | Diagnosis, cloning, synthesis search, scoring, and injection exist; historical replay and measured causal comparison do not. |
| Ingestion | [`src/ingest/`](../src/ingest/), [`src/benchmarks/mscore.py`](../src/benchmarks/mscore.py) | Flexible local parsing and normalization; not a complete official benchmark execution/evaluation stack. |
| Telemetry | [`src/streaming/`](../src/streaming/) | Tested service interfaces and buffers; missing automatic reasoning-process bridge and trustworthy full run provenance. |
| Visualization | [`frontend/`](../frontend/), [`visualizer.html`](../visualizer.html) | React dashboard and separate HTML trace player; substantial static/demo behavior remains. |
| Analysis | [`src/analytics/`](../src/analytics/) | Working aggregation/plotting utilities for supplied records; conclusions inherit the validity of their inputs. |
| Packaging/CI | [`pyproject.toml`](../pyproject.toml), [workflow](../.github/workflows/test.yml) | Central dependency declaration, source layout, Python tests, and schema comparison. Broader integration/build checks absent. |

### Separate execution paths matter

1. **Agent smoke command:** `scripts.s2_smoke` uses `PEXAgent` and `InMemoryBoard`. It does not run the shared scheduler and historical recovery pipeline. It exports a trace after deliberation.
2. **Fixture recovery command:** `fixtures.spike_starter` loads recorded disagreement, uses a scripted mock answer, and injects a revision. It demonstrates mechanics, not newly measured model recovery.
3. **Default benchmark path:** `BatchTrialRunner` generates probabilistic simulation records. Increasing its trial count increases simulation samples, not evidence from actual agent deliberation.
4. **Live benchmark executor:** uses model-backed agent turns but its own simplified loop, a task-ID-only prompt, and shortcut recovery. It does not invoke the main scheduler/solver integration.
5. **Telemetry server:** receives events/snapshots explicitly. Starting it does not subscribe to scheduler Redis publications automatically.
6. **HTML visualizer:** plays built-in scenarios or a fetched/injected JSON trace. It is distinct from the React WebSocket dashboard.

These paths should be described separately in demonstrations until a shared integrated path is verified.

## 4. What is working and worth retaining

### Contracts and packaging

The Pydantic contracts provide a useful common vocabulary: prediction/explanation payloads, contributions, snapshots, scheduled turns, locks, recovery proposals, events, task configuration, and results. All 14 tracked schema exports matched in a fresh temporary export comparison.

The `src/` layout and editable installation are documented. Package data includes personas, fixtures, and schema stubs. This is a sound organizational foundation, although this review did not rebuild/install a wheel in a fresh environment.

### Model broker and agent development path

The broker offers synchronous calls, futures, an async wrapper, a semaphore limiting concurrent calls, retries/backoff, and thread-protected usage counters. Mock tests cover concurrency limits, retries, async calls, and accounting.

The agent implementation includes persona instructions, recent board context, schema-constrained generation requests, bounded parsing repairs, contribution construction, and a stale-turn retry against the dedicated board. The parser tolerates several common formatting errors. Parallel generation preserves agent ordering while sharing an input snapshot.

These are useful capabilities. They do not yet establish hardware-memory safety, provider reliability, or valid semantic agreement.

### Scheduling, diagnosis, and sandbox primitives

The scheduler has an explicit priority queue, multiple arbitration policies, store commit hooks, consensus/deadlock heuristics, and recovery callback registration. Fixture tests exercise controlled turn dispatch and recovery injection.

Attribution traces parent links with cycle protection, collects conflicting agents/evidence, and identifies ratified checkpoints. Sandbox sessions clone bounded history windows, retain simulated entries, support checkpoint escalation, and export graph-shaped telemetry. These operations provide a practical base for more faithful replay.

### Service, exports, and analysis

FastAPI provides health, session history, snapshots, graph extraction, event ingestion, and global/session WebSockets. Existing tests verify several API/WebSocket flows. The manager bounds per-session history and uses per-socket locks during broadcast sends.

Dataset adapters accept local JSON/JSONL inputs, normalize several aliases, and produce shared task/configuration objects. Aggregation and plotting utilities produce usable tables and figures from `TrialResult` records. The saved-results README explicitly labels historical examples as simulations, which is appropriate.

## 5. Reproduced failures outside the existing test assertions

The following probes used current repository classes with offline inputs. Monkeypatches were process-local; they did not modify implementation files. The repeated live-executor probe used a mock backend and disabled trace file export.

| Probe | Observed result | Consequence |
| --- | --- | --- |
| Register a real `PEXAgent` directly in the scheduler | `TypeError: Agent handler for general_reasoner_0 is neither callable nor an agent instance.` | The advertised agent and scheduler interfaces do not directly connect. |
| Wrap `agent.act()` as a scheduler callable using `InMemoryBoard` | `StaleTurnError: turn_index 1 != expected 0` | An adapter alone does not resolve the incompatible turn convention. |
| Render history after adding a flat entry to `InMemoryBlackboard` | `AttributeError: 'BlackboardEntry' object has no attribute 'target_contribution_id'` | Its declared snapshot and actual stored history disagree. |
| Ask scheduler consensus predicate about distinct agents ratifying `A` and `B` | Returned `True` | Trailing ratification tags can certify incompatible positions. |
| Redis-style write with another session/agent, index 99, and missing target | Accepted `other_session`, agent `other`, turn 99 into session `s` | A valid lock does not guarantee a valid contribution. |
| Parse valid nested model prediction/explanation through simulator sanitation | Actual content replaced by `FALLBACK`, confidence 0.85, empty evidence | Valid production-shaped model output can become synthetic content. |
| Render an MSCoRe snapshot for an agent | Reference answer appears in the prompt | Hidden evaluation information is exposed to inference. |
| Import `BatchTrialRunner` and supply a custom executor | `NameError: name 'asyncio' is not defined` | The imported extension path fails; CLI execution masks the missing module-level import. |
| Simulate with allowed `max_turns=2` | `ValueError: empty range in randint(4, 2)` | Valid configuration values are not supported by the implementation. |
| Simulate task `probe`, seed 1, zero capability density | One counterfactual intervention recorded | The no-recovery control can contain recovery. |
| Execute the same two-turn mocked live configuration twice | Token totals **1,179**, then **2,358** | Each trial receives cumulative broker usage rather than its own usage. These are mock token estimates, not live model measurements. |
| Inspect first live-executor prompt for task `probe` | Contains `Benchmark Problem: probe`, context `{"domain": "general"}` | The actual dataset question/context never reaches the agent through this factory. |

These observations are stronger evidence than historical completion percentages. They also explain why 179 green tests and serious integration defects can coexist.

## 6. Findings by subsystem

### 6.1 Shared board contracts and storage

**Interface divergence.** `InMemoryBoard` expects zero-based turn indices and nested contributions. The scheduler calculates a one-based step in relevant dispatch paths. `InMemoryBlackboard` appends flat entries into a history declared as contributions, with no equivalent authoritative validation. Redis exposes a lock/write API rather than the agent's `submit()` contract.

**Information loss.** Converting a contribution through a flat `BlackboardEntry` does not preserve all instrumentation fields and counterfactual semantics. The local blackboard does not maintain leading claim, active agents, and version in parity with the other board. Retaining caller-owned mutable objects also weakens isolation of committed history.

**Atomicity risk from source inspection.** Redis lock ownership and version are checked before a separate read/modify/save sequence. If a lease expires between validation and persistence, a stale writer can overwrite a newer state. `set_status()` independently reads and rewrites the whole snapshot. This fresh review identifies the vulnerable sequence; it did not reproduce that interleaving against a real Redis server.

**Graph integrity.** Redis sandbox cloning renames contribution IDs but does not consistently remap target references or contribution session IDs. It also retains the source's active claim even when truncating history. A clone can therefore have dangling references or state derived from later history.

**Improvement opportunities:** define one canonical submission contract, turn convention, input-copy rule, and snapshot invariants; validate identities/targets/duplicates/branches at the authoritative boundary; make complete state transitions atomic; reconstruct consistent historical state when cloning. These are proposed improvements, not completed fixes.

### 6.2 Scheduling and protocol meaning

Consensus checks trailing `RATIFY` tags and distinct agents without checking that they agree on the same claim and explanation. Deadlock checks a short disagreement sequence; useful dissent can meet that heuristic. Different modules use different windows and success rules.

Capability is represented as the agent role `COUNTERFACTUAL`, replacing its ordinary domain role. Some policies exclude that role as a one-shot mediator. Enabling a capability can therefore change who participates in ordinary deliberation, confounding the intended density comparison.

Recovery injection can replace an existing agent's normal handler with a closure that repeatedly returns the revision. Reactive scheduling can also include a registered mediator in later ordinary turns. Pending targets are not comprehensively revalidated following recovery.

`run()` checks some terminal states, but direct `step()` does not provide equivalent terminal protection. Turn-budget exhaustion can leave an active status. Handler/commit errors can leave stale active-agent bookkeeping.

**Improvement opportunities:** anchor votes to an explicit shared position; distinguish capability from role; apply revisions once; preserve normal handlers; define terminal/budget/failure outcomes and enforce them at all entry points. The quorum and adapted PXP semantics require a documented team decision, rather than an arbitrary threshold change.

### 6.3 Agent output, prompts, and inference

The parser/repair layer is useful, but subsequent semantic normalization can manufacture agreement: an inconsistent `RATIFY` claim is overwritten with the target's claim, and an invalid target may be redirected to an unrelated contribution. A schema-valid repaired object is not necessarily the agent's actual judgment.

Prompt context is bounded by character counts and a fixed history length, rather than a measured token budget preserving target dependencies. Global broker accounting works for a broker lifetime, but callers must measure per-trial deltas or use an explicit call ledger. The live runner does neither.

The semaphore caps simultaneous calls; it does not measure whether several loaded models fit available RAM/VRAM. Several environment settings are unvalidated or not consistently passed through all paths; for example, temperature in the example environment file is not wired by the broker factory into all agent construction.

**Improvement opportunities:** bounded semantic re-prompting instead of silently changing meaning; explicit failure records; token-aware context selection; per-call/per-session accounting; validated configuration and measured provider readiness. Actual Ollama/LiteLLM compatibility remains a separate runtime validation task.

### 6.4 Counterfactual recovery and scoring

The current algorithm synthesizes a new compromise and asks a generated opposing peer for one reaction. It does not replace one historical assertion and replay later turns through the original roster/scheduler alongside an unchanged baseline. Structural tracing and tag-based blame are diagnostic heuristics, not measured causal effects.

The simulator and daemon silently fall back to mocks when no broker is supplied. Invalid or absent peer tags become `RATIFY`. Nested fields from the agent output schema are discarded by a sanitizer expecting strings. The production request adaptation does not reuse the full persona/parser contract used by ordinary agents.

Search candidates append to the same sandbox history. The final scorer sees accumulated simulation entries, while the selected revision can come from one best candidate. A score, trajectory, and proposed revision can therefore refer to different units of evaluation. The simulator and final scorer also use different scoring objectives and convergence conditions.

Ground truth can affect agreement scoring. Explanation length, lexical overlap, tag entropy, and self-reported confidence are treated as quality proxies without independent factual validation. Evidence overlap does not detect negation or incorrect interpretation. Fixed blame/credit weights do not establish that a critic caused an incorrect outcome; a correct minority critic can receive blame for disagreeing.

The solver emits `COUNTERFACTUAL_RESOLVED` when proposing a candidate, before live ratification. Cleanup relies on external calls, and sessions/recovery histories can remain retained after episodes.

**Improvement opportunities:** explicit execution mode; strict shared parsing; isolated candidate checkpoints; historical baseline/intervention replay if that remains the research objective; label-free selection; consistent provenance; fresh live confirmation; bounded shared inference/time budgets and cleanup on all exits. Prospective mediation could instead be studied, but it should be named and evaluated as that method.

### 6.5 Dataset ingestion and benchmark execution

Adapters are parsing infrastructure. They do not implement the full task environments or independent evaluators needed to establish official benchmark results. No FHIR action executor, data-lake execution harness, or common tool observation/action lifecycle was found in the current reasoning path.

MSCoRe's alias parser does not handle an `instruction`/`input`/`output` record as a complete task. KramaBench parsing does not map `answer`/`answer_type`; nested subtasks are copied into agent context without removing answer fields. This review identifies the schema coverage and exposure behavior, not the current official dataset specification. Dataset releases and permitted input fields still need to be pinned and verified externally.

Truthiness-based alias selection can lose valid zero/false answers. Resampling changes IDs and role metadata, but that variation is not consistently preserved in execution configurations. Renaming repeated sample questions and adding unrelated domain labels does not create new independent benchmark tasks.

The imported batch runner fails on `asyncio`; the live executor uses the task ID, ignores configured model mappings in favor of its factory settings, and bypasses scheduler/solver recovery. Extra ordinary turns are counted as recovery, and a `REVISE` can terminate with `RESOLVED`. It rewrites the returned top-level session ID without consistently rewriting nested assessment/trace identities.

The simulation encodes a density-dependent recovery probability, assigns intelligibility automatically, estimates costs, and can report interventions at zero density. Such records can demonstrate analysis formats, but cannot test whether counterfactual capability improves real reasoning. Trial-count rounding, tiny budgets, and synthetic tag-count consistency also need attention.

**Improvement opportunities:** resolve configurations to actual tasks; separate evaluator-only fields; select and document official or explicitly adapted execution paths; use shared deliberation/recovery; measure costs and correctness independently of consensus; persist failures, seeds, manifests, and incremental results. Freeze study design only after a small valid pilot, rather than scaling the current simulation.

### 6.6 Telemetry and service behavior

The store publishes to Redis Pub/Sub channels; FastAPI uses its own in-process singleton. There is no implemented subscriber bridging those channels into that singleton. Importing a singleton in another Python process does not share state.

Store events often include only summaries rather than the full contribution/snapshot needed to reconstruct the UI. The daemon emits limited resolved metadata but does not supply the full branch trajectory and credit lifecycle expected by the frontend. Deadlock field names also differ between producer and consumer (`deadlock_turn_index` versus `turn`).

History is process-local and disappears on restart. Slow WebSocket sends have no explicit timeout; bounded session-history policy can exceed its target if sessions remain actively subscribed. Initial/control WebSocket sends bypass the per-socket broadcast lock. Snapshot updates do not validate agreement between URL session ID and body session ID.

The service exposes event ingestion and snapshot mutation without authentication and allows broad CORS. This is a concrete boundary to address if the team exposes it beyond local demonstrations; it does not prevent local offline research use.

**Improvement opportunities:** one event schema with complete provenance, an explicit cross-process bridge, durable ordered history where required, reconnect recovery, consistent payloads, send timeouts, and an intentional deployment access model.

### 6.7 Dashboard and HTML trace player

React components can render nested or flat contribution fields, and the hook recognizes several recovery event names. However, the reasoning graph is a sequence of cards connected in chronological order, not a graph built from actual target relationships. The installed React Flow dependency is not used by that component.

The ablation dashboard contains fixed sample metrics. The Run button's callback only logs to the console. Lock state defaults to `IDLE`, and the active-agent display can default to three agents instead of reporting missing data.

WebSocket cleanup closes the socket, whose `onclose` then schedules reconnect. Without an intentional-close flag and stale-callback guards, unmount/session changes can reopen old connections; React StrictMode makes lifecycle correctness particularly important. Session changes do not clear all previous state, events grow without a frontend bound, and available-session discovery is primarily populated at connection time.

Demo fallback can remain visible while the socket is connected but has not supplied events. The hardcoded `ws://...:8000` URL bypasses the configured Vite proxy and does not accommodate HTTPS `wss://` deployments. Playback slices the graph but leaves other panels showing full/current history.

The HTML player fetches `outputs/live_trace.json` or accepts injected trace data. Its visual detail does not establish that sandbox candidates were actually executed.

**Improvement opportunities:** explicit demo/live state; real job/results wiring or an honestly disabled action; target-based graph edges; session isolation; correct reconnect lifecycle; bounded event state; one playback cursor; configurable secure endpoints. These UI findings are based on source inspection, not a fresh browser test.

### 6.8 Trace authenticity and presentation claims

This is a high-impact research-reporting issue. `export_snapshot()` inserts fixed sandbox trials with scores **0.54, 0.78, and 0.95**, including simulated prompts, responses, critiques, and winner claims. Those are not read from measured solver trajectories. It can describe counterfactual-marked output as resolved without confirmed live agreement, and infer a consensus summary from recent ratification even when the authoritative final status differs.

`export_from_spike_blackboard()` uses arithmetic formulas for token counts and latency, a fixed step for deadlock, and hardcoded GPU/model labels. The spike itself uses a queued mock answer. These generated numbers must not be presented as real Qwen/Ollama measurements.

Archived presentation notes describe successful model/sandbox behavior more strongly than the current execution path supports. Archiving preserves that history but does not make those claims accurate.

**Improvement opportunities:** export recorded events/calls/candidates only; keep unknown measurements unknown; clearly mark scripted examples; preserve authoritative final status and hardware identity; attach trace provenance and source run IDs.

## 7. Existing results and scientific interpretation

| Artifact | Fresh observation | Appropriate use |
| --- | --- | --- |
| `examples/benchmark-results/results.json` | 500 records; all inspected notes identify simulation; one intervention appears in the zero-density subset. | Historical output format and plotting example. |
| `outputs/results.json` | 8 local records; all notes identify simulation; zero interventions in that collection's zero-density subset. | Local simulation example; ignored generated output. |
| Dashboard sample table | Hardcoded rates and costs. | UI demonstration only. |
| Exported sandbox score narrative | Fixed template values in exporter source. | Scripted illustration, not measured recovery evidence. |

The analytics library summarizes the records it receives; it does not independently establish whether they were produced by a valid experiment. No evidence here establishes a measured improvement in task correctness, measured causal blame, or knowledge transfer.

Consensus and correctness need separate records: agents can unanimously agree on a wrong answer. Formal Strong/Ultra-Strong intelligibility cannot be inferred merely from a capability percentage or the presence of a revision. Local research notes also contain numerical failure distributions without sufficient traceable support in the documents; treat them as unverified until checked against primary evidence.

## 8. Testing, maintainability, and repository health

### Useful existing foundations

- Central Python dependency configuration and source/package-data declarations.
- Tests for parser behavior, concurrency caps, mock storage locking, scheduling, attribution, sandbox isolation, scoring, exports, API/WebSocket behavior, and analytics.
- CI configured for two supported Python versions and temporary schema comparison.
- A separate ignored output directory and preserved simulation examples with an explanatory README.
- Clear module boundaries that can support focused repairs without a wholesale rewrite.

### Gaps in current verification

- No test demonstrating the actual `PEXAgent` class completing a scheduler-driven multi-turn session through both supported storage arrangements.
- No real Redis service, lease-expiry race validation, or authoritative-write adversarial cases in CI.
- No frontend build/browser checks in the current workflow.
- No paired historical replay, hidden-label independence, strict malformed recovery failure, or complete live-confirmation proof.
- No integrated benchmark evaluator, real dataset/environment smoke, restart/resume, or cross-process telemetry proof.
- Some tests validate permissive fallback and label-influenced behavior that should be reconsidered if the research requirements reject it.

### Documentation state after this cleanup

Nine old planning/review/mixed documents now live under [`docs/archive/plans/`](archive/plans/README.md), marked inactive. Earlier completed/unstarted/partial classifications are retained for reference. No whole historical plan was marked fully complete.

The root README remains the setup entry point and now links here. Research references, the runnable notebook, demo commands, and Student 4 setup guide remain accessible. Some surviving historical setup/reference prose may still contain stale claims; it should not override tested current behavior.

The scoped ignore exception makes archived local plans eligible for Git tracking. It does not stage, commit, or publish them. No Git history or remote branches were changed.

## 9. Improvement priorities for discussion

These are review recommendations, not a replacement timetable. They are grouped by consequence so the team can create one future plan after agreeing on the outcome it wants.

| Priority | Improvement area | Why it matters | Evidence of success to look for |
| --- | --- | --- | --- |
| Critical | Honest traces and evaluator isolation | Synthetic measurements and answer leakage invalidate conclusions. | An exported trace reconciles to actual calls/events; changing hidden answers cannot alter inference or selection. |
| Critical | Shared agent/board/scheduler contract | Currently prevents the real components from executing together. | Same agent class runs multiple scheduled turns with canonical snapshots and consistent indices. |
| Critical | Authoritative writes and shared-position consensus | Invalid writes and conflicting votes can corrupt or prematurely end reasoning. | Bad identities/targets/stale writes are rejected; incompatible ratifications cannot terminate successfully. |
| High | Explicit, faithful recovery | Current synthesis does not establish the promised historical effect. | Isolated baseline and intervention histories, shared budgets, one-time application, and fresh live confirmation. |
| High | Actual task execution and independent grading | Current live benchmark lacks the question and valid recovery/evaluation. | Real task reaches prompts; final answer is independently scored; zero capability performs zero recovery. |
| High | Per-trial accounting and durable provenance | Cumulative usage and template artifacts prevent cost/reproducibility claims. | Manifest, task/seed IDs, call ledger, measured/unknown usage, and incremental outcomes reconcile. |
| Medium | Connected, accurate telemetry/UI | Dashboard currently cannot reliably represent the separate reasoning process. | Separate producer/server display the same ordered history and branch state; reconnect/session tests pass. |
| Medium | Broader meaningful integration checks | Component tests currently miss the most consequential failures. | CI exercises canonical integration, real Redis, frontend build, and selected browser lifecycle cases. |
| Later | Model diversity, routing, and larger studies | Adds variables and compute before the current method is validated. | Consider only after a small valid baseline/recovery/evaluation pilot establishes actual cost and behavior. |

## 10. Decisions still needed before a new implementation plan

The team should settle the following questions using this review as a baseline:

1. Is the research method historical intervention/replay, prospective mediation, or a comparison of both?
2. What exactly constitutes agreement on both prediction and explanation, and which agents must ratify?
3. Which benchmarks will use their official environments/evaluators, and which will be explicitly adapted subsets?
4. What is held constant across capability conditions: personas, model, seeds, tool access, context, and inference budgets?
5. Which outcomes must be measured separately: correctness, consensus, recovery attempts, confirmed recovery, tokens, runtime, and failures?
6. What trace and pilot evidence is required before calling the project integrated or launching a larger study?

No individual ownership, delivery dates, completion percentages, or final study size are assigned here. The current code provides enough foundation to improve incrementally, but the next plan should be anchored to demonstrated execution and evaluation rather than the historical milestone claims.

## 11. Changes made for this review

- Moved the previously collected plans to an explicit inactive archive, preserving their contents except relocation links.
- Added the archive landing page and this status review.
- Updated README navigation and the scoped archive ignore exception.
- Re-ran existing backend tests, compared exported schemas in temporary storage, and ran process-local offline probes.
- Made **no implementation, test, dependency, notebook, or benchmark-result changes**. Nothing was staged, committed, pushed, or deployed.
