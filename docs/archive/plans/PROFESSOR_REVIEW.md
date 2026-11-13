# Professor's repository review

Review date: 2 October 2026. Reviewed working tree: commit `7ec4e2f` plus the existing local changes to `data/results.*` and the three figures. This review does not change the implementation or experiment outputs.

## Overall assessment

The repository contains substantial work from all four students: shared contracts, storage and scheduling, a model broker and persona agents, a bounded sandbox and recovery solver, streaming endpoints, a React dashboard, dataset adapters, tests, and plotting utilities. The code is organized well enough to support further integration.

**My assessment is that this is a developed prototype with important integration and experimental-validity gaps. I would not yet accept it as a completed, empirically validated implementation of the proposal.** Passing the current tests establishes useful component behavior; it does not establish that real agents, historical counterfactual replay, official benchmark scoring, and the live dashboard function as one system.

The most urgent work is to repair the integration contracts and establish a valid experiment. Additional visual polish or larger simulated batches will not resolve those gaps.

## Basis, ownership, and category definitions

The PDF explicitly supersedes the proposal's original three-person allocation. Its revised responsibilities on pages 3–5 are the basis for ownership below. In particular, **Student 1 owns both KramaBench and MedAgentBench adapters**; Student 4 owns MSCoRe and the shared runner. The PDF's earlier summary mentions only the MedAgentBench transfer, but its detailed responsibility list and dependency map assign both adapters to Student 1.

| Student | Assigned responsibility | Work reviewed | Assessment against the plan |
| --- | --- | --- | --- |
| 1 | Infrastructure and protocol controller | `contracts/`, `storage/`, `blackboard/`, `scheduler/`, `ingest/`, corresponding tests | Broad component coverage; transactional safety, shared board behavior, protocol enforcement, and official-data fidelity need correction. |
| 2 | Agent engineering and inference | `agents/`, `llm_broker/`, `prompts/`, smoke script, documentation and tests | Useful broker, personas, parsing and instrumentation; integration with the scheduler, capability semantics, and live-model evidence remain incomplete. |
| 3 | Counterfactual sandbox and deadlock recovery | `sandbox/`, `counterfactual/`, spike fixture, scoring guide and corresponding tests | Substantial sandbox and recovery machinery; current search is mainly synthesis and peer reaction, with heuristic attribution rather than measured retrospective causal credit. |
| 4 | Visualization and benchmarking | `streaming/`, `frontend/`, `benchmarks/`, `analytics/`, results, figures, documentation and tests | Useful service and dashboard scaffolding; live connectivity and evaluation need major corrections before the outputs support research claims. |

Responsibility labels identify who should lead a correction. They are not a claim about individual authorship or effort: shared interfaces require joint fixes, and the plan does not identify students by personal name.

The three categories mean:

- **Must change:** a demonstrated defect, a missing core commitment, or an issue that invalidates conclusions. Resolve these before claiming the affected milestone or publishing the ablation results.
- **Recommended to change:** improvements to reliability, reproducibility, maintainability, testing, or presentation after the core blockers are fixed.
- **Change should be reviewed before applying:** choices that alter protocol semantics, research scope, experimental controls, or compute requirements. Agree on them with the supervisor and affected students before changing the implementation.

## Verification and limits

- Full existing suite: **172 passed, 1 warning, in 2.36 seconds**, using the existing virtual environment and `.venv/bin/python -m pytest -q -p no:cacheprovider` outside the sandbox. The initial streaming timeout disappeared outside the sandbox; it is not counted as a repository defect.
- All **14 exported JSON schema stubs** match the current Pydantic schemas when compared as JSON.
- Separate offline probes used the actual repository classes with `MockBackend` and deterministic input histories to reproduce the failures listed below. A controlled lease-expiry interleaving reproduced a lost write; this was not a stress test against a real Redis server.
- The working tree's JSON and CSV each contain **4 simulated results**, one per density, for a single repeated sample question. The committed `HEAD` JSON contains **500 simulated results**, all labeled MSCoRe, including one counterfactual intervention at zero density. These are distinct observations: the local four-row files were already modified when review started.
- No real 7B/8B inference, official dataset evaluation, or browser build was executed. Node/npm and frontend dependencies are absent in this environment. Frontend conclusions below are from source inspection, not an executed browser session. The local Python environment is 3.14.7; this run does not establish the workflow's 3.11/3.12 compatibility.
- The proposal, revised PDF, README files, overview, reference notes, previous review document, scoring guide, current implementation, relevant tests, CI configuration, and current/committed results informed the review. Older review claims were not assumed to describe the current code.

## Category 1: Must change elements

### Student 1 — Infrastructure and protocol controller

**S1-M1. Establish one working board/agent/scheduler contract.** Joint owners: S1 and S2.

Evidence: [scheduler handler dispatch](/home/link4444/Downloads/Ai/project_blackboard/scheduler/engine.py:260), [agent turn construction](/home/link4444/Downloads/Ai/project_blackboard/agents/base.py:151), [stub board submission](/home/link4444/Downloads/Ai/project_blackboard/agents/board_client.py:67), [in-memory board submission](/home/link4444/Downloads/Ai/project_blackboard/blackboard/store.py:65), and [entry conversions](/home/link4444/Downloads/Ai/project_blackboard/contracts/schemas.py:878).

The scheduler accepts a callable or an object exposing `generate_turn`; the real `PEXAgent` exposes `act` and `step`. Registering a real agent directly raises `TypeError`. A thin callable around `act` then hits a second problem: the scheduler changes the first zero-based turn to 1, but `InMemoryBoard` expects 0. The other `InMemoryBlackboard` appends a flat `BlackboardEntry` into a list declared to contain nested `AgentContribution` objects; the second real-agent turn fails while rendering history. Flat conversions also lose token usage, latency, and the counterfactual flag.

Change: define a canonical snapshot/contribution format, turn-index convention, and explicit adapter for real agents. Both supported stores must honor the same submission invariants. Preserve instrumentation across any remaining compatibility conversions.

Acceptance: the same real `PEXAgent` class, backed by an offline model, completes a multi-turn scheduler session through both supported storage paths; typed snapshots round-trip with tokens and flags intact. This is a contract repair, not a requirement to remove every legacy type immediately.

**S1-M2. Make the complete state mutation atomic.**

Evidence: [Redis write](/home/link4444/Downloads/Ai/project_blackboard/storage/redis_store.py:441) and [status update](/home/link4444/Downloads/Ai/project_blackboard/storage/redis_store.py:539).

Lock ownership and expected version are checked separately from saving state. If the lease expires after those checks, a new writer can commit and then have its update overwritten by the stale writer. The review reproduced two successful commits leaving version 1 and only the first writer's contribution. `set_status` also performs an unguarded read-modify-write of the entire snapshot, which can overwrite a concurrent append.

Change: perform lock verification, version comparison, append, and version/state update in one server-side transaction or atomic script. Apply equivalent protection to status mutations. The in-memory implementation needs a lock covering the full operation, not just individual Redis-like commands.

Acceptance: a lease-expiry interleaving rejects the stale commit and preserves the new writer's data; simultaneous status and contribution changes preserve both. Real Redis verification is recommended below.

**S1-M3. Enforce protocol and identity invariants at the authoritative write boundary.**

Evidence: [Redis append](/home/link4444/Downloads/Ai/project_blackboard/storage/redis_store.py:471), [canonical contribution schema](/home/link4444/Downloads/Ai/project_blackboard/contracts/schemas.py:190), and [local append](/home/link4444/Downloads/Ai/project_blackboard/blackboard/store.py:42).

A locked Redis write accepted a contribution belonging to another session, a `RATIFY` targeting a missing ID, and turn index 99 on an empty board. Duplicate IDs, invalid target relationships, and PXP content consistency are not centrally checked. Schema-valid tags alone do not implement the promised state controller. The local store also retains the caller's mutable entry object.

Change: validate session and agent identity, unique IDs, sequential turns, allowable target/branch relationships, non-empty PEX content, and the agreed simplified protocol before mutation. Reject invalid writes consistently across stores; copy committed inputs to prevent later caller mutation.

Acceptance: invalid relationships and duplicates are rejected, legitimate opening revisions are accepted, and mutating a previously submitted object cannot change stored history.

**S1-M4. Require agreement on a shared PEX position and enforce terminal behavior.**

Evidence: [consensus predicate](/home/link4444/Downloads/Ai/project_blackboard/scheduler/engine.py:322), [run loop](/home/link4444/Downloads/Ai/project_blackboard/scheduler/engine.py:447), and [in-memory state handling](/home/link4444/Downloads/Ai/project_blackboard/blackboard/store.py:114).

The consensus predicate checks trailing `RATIFY` tags and agent diversity, without checking that those agents ratify the same claim and explanation. The review passed it two conflicting ratifications and received `True`. The in-memory store does not consistently update the leading claim/version. Reaching the turn limit leaves an active session, and direct `step()` calls do not enforce a terminal guard.

Change: anchor ratification to a claim/explanation revision or explicit consensus object, update leading state consistently, and record budget exhaustion as a defined terminal outcome. Prevent submissions/dispatch after termination unless the session is explicitly reopened.

Acceptance: ratifications of incompatible positions cannot end the session; budget exhaustion produces a distinct recorded outcome; terminal sessions do not execute additional turns. Choice of quorum is a review-before-applying decision below.

**S1-M5. Apply recovery as a one-time turn without replacing a normal agent's handler.**

Evidence: [recovery injection](/home/link4444/Downloads/Ai/project_blackboard/scheduler/engine.py:398).

When a proposal names an already registered agent, recovery permanently replaces that agent's handler with a closure returning the same revision. The review confirmed the original handler is not preserved. Pending turns also retain targets chosen before recovery, so the policy can dispatch outdated work after the board changes.

Change: represent a recovery payload as a one-time scheduled action, preserve the normal handler, and cancel or revalidate obsolete queued turns against the new board version. Distinguish one-time mediators from ordinary agents with extra capability.

Acceptance: subsequent normal turns still call the original agent; the recovery executes once; every dispatched target is valid for the current hypothesis.

**S1-M6. Correct KramaBench parsing and remove hidden subtask answers from agent context.** Joint owner: S4.

Evidence: [reference-answer parsing](/home/link4444/Downloads/Ai/project_blackboard/ingest/kramabench.py:239) and [subtask context](/home/link4444/Downloads/Ai/project_blackboard/ingest/kramabench.py:257). Official workload entries use `answer` and `answer_type`; the adapter does not map those fields. A representative record lost both the reference answer and grading type. It also copied a subtask's `answer` into the agent-visible context. The official workload demonstrates these fields. [Official KramaBench workload](https://raw.githubusercontent.com/mitdbg/Kramabench/main/workload/legal.json)

Change: retain reference values, types, and evaluator metadata in a separate evaluation record. Expose only permitted inputs to agents, including a sanitized task decomposition if the experiment allows it. Do not pass reference outputs or solution recipes through context.

Acceptance: fixtures matching a pinned official release retain numeric/string grading information, and rendered agent prompts contain no main-task or subtask gold answers.

**S1-M7. Preserve graph identities and historical state during Redis sandbox cloning.**

Evidence: [Redis clone](/home/link4444/Downloads/Ai/project_blackboard/storage/redis_store.py:558).

The clone renames contribution IDs with `_sbx` but leaves target IDs and contribution session IDs pointing to the source. A cloned two-turn history had a dangling parent and both contributions still belonged to the original session. A rollback also copies the source's current active claim, even if that claim was established after the rollback point.

Change: choose a consistent identity strategy, remap relationships where IDs change, bind contributions to the new session, and rebuild leading state from the retained history.

Acceptance: every internal target resolves within the cloned graph; all session references are consistent; rolling back cannot retain future knowledge.

### Student 2 — Agent engineering and inference

**S2-M1. Separate domain role from counterfactual capability.** Joint owners: S1 and S3.

Evidence: [agent role](/home/link4444/Downloads/Ai/project_blackboard/agents/base.py:71), [panel construction](/home/link4444/Downloads/Ai/project_blackboard/agents/panel.py:24), and [round-robin/priority eligibility](/home/link4444/Downloads/Ai/project_blackboard/scheduler/policies.py:65).

Enabling counterfactual capability changes an agent's role to `COUNTERFACTUAL`. Scheduler policies treat that role as a one-time mediator and exclude it from ordinary turns whenever ordinary agents remain. With two capable agents in a three-agent panel, the review observed round-robin selecting only the remaining critic on every turn. The ablation therefore changes ordinary participation and expert-role availability as well as capability.

Change: keep `PRIMARY`, `CRITIC`, and domain-expert roles stable. Store capability separately and identify one-time recovery actions explicitly. Only the additional retrospective operation should vary across density conditions.

Acceptance: all three agents retain their ordinary roles and participate at 0%, 33%, 66%, and 100%; changing capability alone does not remove experts from the schedule.

**S2-M2. Stop silently converting semantic disagreement into agreement.**

Evidence: [tag normalization](/home/link4444/Downloads/Ai/project_blackboard/agents/base.py:113) and [existing test](/home/link4444/Downloads/Ai/project_blackboard/tests/test_s2_agents.py:132).

If a model emits `RATIFY` with a different claim, the code replaces its claim with the target's claim and preserves the rest of its explanation. The review observed `Answer B` becoming `Answer A`. An invalid target can similarly be redirected to an unrelated recent post. Those operations can produce apparent agreement the model did not express. The current test explicitly rewards this behavior.

Change: separate harmless formatting repair from semantic repair. Re-prompt on claim/tag/explanation inconsistency or ambiguous targets, preserving the raw output and repair reason. A bounded failure should remain a failure when no coherent response is produced.

Acceptance: incompatible `RATIFY` output does not create a successful ratification; explanations support the selected claim; ambiguous targets are resolved by the agent or rejected. Record any corrections so results can be audited.

**S2-M3. Supply real-model validation before accepting the inference milestone.**

Evidence: [open live-smoke checklist](/home/link4444/Downloads/Ai/project_blackboard/docs/STUDENT2.md), [smoke script](/home/link4444/Downloads/Ai/project_blackboard/scripts/s2_smoke.py), and [backend implementations](/home/link4444/Downloads/Ai/project_blackboard/llm_broker/backends/ollama.py).

Offline tests establish broker mechanics and scripted parsing. They do not establish that the chosen local model reliably follows the schema or that the complete scheduler can use it. The documentation itself leaves the real 7B check open.

Change: record a small, reproducible real-model run after S1-M1 is fixed. Record model identifier/quantization, backend version, hardware, decoding settings, valid-output rate before/after repairs, final failures, token counts, and latency. Keep raw responses and a task manifest.

Acceptance: at least one planned local model completes a small integrated panel run on real task prompts with bounded errors. This is required evidence for the planned milestone, not a claim that the backend is already proven defective.

### Student 3 — Counterfactual sandbox and recovery

**S3-M1. Parse the actual S2 schema and treat invalid inference as failure.** Joint owner: S2.

Evidence: [broker-output extraction](/home/link4444/Downloads/Ai/project_blackboard/counterfactual/simulator.py:460), [sanitization](/home/link4444/Downloads/Ai/project_blackboard/counterfactual/simulator.py:532), [peer tag fallback](/home/link4444/Downloads/Ai/project_blackboard/counterfactual/simulator.py:326), and [implicit mock fallback](/home/link4444/Downloads/Ai/project_blackboard/counterfactual/solver.py:211).

S2 emits nested prediction/explanation objects, but the sanitizer expects flat strings and top-level confidence/evidence. A valid S2 model claim was replaced with a fallback claim in the review. Non-JSON text defaults to `RATIFY`, and an invalid peer tag also becomes `RATIFY`. A solver with no broker silently uses a scripted broker.

Change: reuse S2's validated decision parser and normalize into the shared contribution contract. Use explicit mock mode only. Exhausted repairs or missing production dependencies must yield a recorded failed branch, not fabricated agreement.

Acceptance: nested S2 output survives intact; malformed responses cannot increase agreement or create a successful branch; production recovery cannot silently switch to fixtures.

**S3-M2. Implement the promised historical intervention and faithful replay, or obtain approval to change the research scope.**

Evidence: [solver's reconstructed board](/home/link4444/Downloads/Ai/project_blackboard/counterfactual/solver.py:185), [synthesis prompt](/home/link4444/Downloads/Ai/project_blackboard/counterfactual/simulator.py:170), and [simulation loop](/home/link4444/Downloads/Ai/project_blackboard/counterfactual/simulator.py:260).

The current loop appends a synthesized revision and one opposing reaction after the cloned divergence history. It does not replace a selected historical assertion and replay the original agents from that point. Generated prompts do not render the cloned task context/history; the solver's reconstructed board loses `initial_context`. Original personas, model choices, and decoding seeds are not faithfully reused. Consequently, successful synthesis does not demonstrate the retrospective intervention promised in the plan.

Change: preserve the task's permitted facts and exact agent configuration, branch immediately before the intervention, substitute the historical output, and regenerate a bounded continuation through the same agent/broker interface. Maintain a comparable baseline continuation. If the team instead intends prospective mediation, revise the objective and claims through the review process described in Category 3.

Acceptance: changing one historical turn changes the replay input while other controls remain fixed; the branch retains original task facts; results distinguish mediation from retrospective replay.

**S3-M3. Make credit a measured counterfactual delta, or label it as a heuristic.**

Evidence: [credit assignment](/home/link4444/Downloads/Ai/project_blackboard/counterfactual/scoring.py:416) and [divergence tracing](/home/link4444/Downloads/Ai/project_blackboard/counterfactual/attribution.py:139).

Blame is currently assigned from fixed tag penalties, confidence, and citation count. It does not compare the outcome with and without the candidate intervention. A parent-link trace identifies a structural ancestor, not causal responsibility; a well-supported critic rejecting a wrong answer can receive substantial blame.

Change: report an operational outcome difference, such as `utility(intervened replay) - utility(baseline replay)`, with the intervention and uncertainty recorded. Keep structural diagnosis as a useful heuristic, but avoid naming its fixed penalty a causal estimate. Remove claims of proven causal attribution unless supported by this comparison.

Acceptance: two cases with similar tag counts but different intervention effects receive different measured credit; justified criticism is not automatically blamed for an incorrect consensus. Supervisor review should settle the exact utility and causal claim.

**S3-M4. Keep alternative candidates isolated and score the selected trajectory.**

Evidence: [candidate append/search](/home/link4444/Downloads/Ai/project_blackboard/counterfactual/simulator.py:279) and [solver branch scoring](/home/link4444/Downloads/Ai/project_blackboard/counterfactual/solver.py:225).

Every search iteration appends another proposal/reaction pair to one sandbox. The simulator tracks its best pair, but the solver scores all simulated entries together and then injects the best pair. The score justifying the injected proposal can therefore describe a mixture of different candidate revisions.

Change: fork each candidate from the same controlled checkpoint, associate its score with its own continuation, and inject only a proposal whose own validated trajectory passes the objective. Keep search attempts as separate diagnostics.

Acceptance: failed candidates do not alter another candidate's baseline or score; the selected proposal's stored trajectory and acceptance score refer to the same branch.

**S3-M5. Remove ground-truth influence from inference-time branch selection.**

Evidence: [ground-truth scoring bonus](/home/link4444/Downloads/Ai/project_blackboard/counterfactual/simulator.py:141) and [search access to labels](/home/link4444/Downloads/Ai/project_blackboard/counterfactual/simulator.py:253).

The search reads the benchmark answer from board state and awards a bonus if it appears in a candidate's text. This uses hidden evaluation knowledge to choose the branch. It can bias recovery success even if the answer was never directly inserted into the model prompt.

Change: keep labels in an evaluator-only record. Search using agent-visible information and the pre-agreed selection objective, then evaluate correctness after the branch/run completes. Explicitly label any separate oracle experiment.

Acceptance: changing only a hidden label cannot change inference prompts, chosen branches, or generated predictions; it can change only the post-run score.

**S3-M6. Emit real branch lifecycle events and distinguish a candidate from live recovery.** Joint owners: S1 and S4.

Evidence: [solver telemetry](/home/link4444/Downloads/Ai/project_blackboard/counterfactual/solver.py:317), [sandbox graph export](/home/link4444/Downloads/Ai/project_blackboard/sandbox/manager.py:131), and [UI event handling](/home/link4444/Downloads/Ai/project_blackboard/frontend/src/hooks/useWebSocket.js).

The solver emits `COUNTERFACTUAL_RESOLVED` when it has created a proposal, before the live board applies and validates it. It does not emit the declared trigger/evaluated branch events with the actual trajectory and credit records needed by the split-panel UI. The frontend clears its deadlock state on that early resolution event.

Change: emit internal events for start, candidate evaluation, selection, application, confirmed recovery, and failed recovery. Include actual branch data and measured/clearly labeled heuristic credit. S4 should own delivery to sockets, consistent with the revised plan.

Acceptance: the dashboard shows the actual explored branch; a proposal alone cannot clear an unresolved live deadlock; failed recovery produces an explicit event.

### Student 4 — Visualization and benchmarking

**S4-M1. Run the actual integrated system on the actual task.** Joint owners: all four students.

Evidence: [live executor](/home/link4444/Downloads/Ai/project_blackboard/benchmarks/mscore.py:414), especially session creation at line 429 and recovery at line 468.

The live executor sends only `Benchmark Problem: <task_id>` and a general domain; it discards the actual question, context, options, source data, and recommended roles. It bypasses the shared scheduler and S3 solver. Its counterfactual intervention is an ordinary extra `agent.step()`. Any resulting `REVISE` is counted as resolved consensus without peer ratification or correctness evaluation.

Change: resolve each configuration to its normalized task, create the canonical board, register the appropriate panel with the shared scheduler, and attach the actual bounded solver only when the ablation condition permits it. Require validated live consensus after recovery.

Acceptance: changing the task question changes the model prompt; the same scheduler/solver used by the demonstration produces the benchmark trajectory; zero density never invokes counterfactual recovery; a revision alone is not success.

**S4-M2. Separate simulation output from empirical results and figures.**

Evidence: [default simulation](/home/link4444/Downloads/Ai/project_blackboard/benchmarks/mscore.py:257), [density-based resolution probability](/home/link4444/Downloads/Ai/project_blackboard/benchmarks/mscore.py:279), and current `data/results.json`.

The default runner samples success using an explicitly density-dependent formula. Token and duration values are estimates, not measured inference. The method is useful for exercising analytics, but its positive density trend is encoded into the generator. It can even record a counterfactual intervention at 0% density; an offline probe reproduced this at seed 1. Existing result notes identify simulations, but file naming and figures lack a strong separation from research artifacts.

Change: require an explicit run mode; add provenance to every result; store demos/simulations separately; visibly label synthetic charts. Regenerate paper figures only from validated live executions. Keep a zero-capability invariant even in simulations.

Acceptance: research-report generation refuses simulated input unless explicitly producing a labeled demonstration; every result identifies its execution mode; baseline interventions are zero.

**S4-M3. Parse official MSCoRe records and execute all promised benchmarks.** Joint owner: S1 for adapter integration.

Evidence: [MSCoRe field mapping](/home/link4444/Downloads/Ai/project_blackboard/benchmarks/mscore.py:90), [agent-visible snapshot](/home/link4444/Downloads/Ai/project_blackboard/benchmarks/mscore.py:168), and [CLI task expansion](/home/link4444/Downloads/Ai/project_blackboard/benchmarks/mscore.py:542).

The source-linked MSCoRe dataset card describes `instruction`, `input`, and `output`; these keys are not mapped. A representative official-format record became an empty question, empty reference answer, and missing context without an error. For records it does parse, `to_blackboard_snapshot` puts `ground_truth` into `initial_context`, and S2's prompt renderer exposes it to the agent. The CLI has no dataset-selection/path workflow: it repeats four handwritten samples with changing domain prefixes and still labels all trials MSCoRe. Adding an automotive title to a medical sample does not create a new benchmark task. [MSCoRe dataset card](https://huggingface.co/datasets/032564yn/MSCoRe/blob/main/README.md)

Change: implement release-specific validated parsing, preserve domain/task type, reject missing required fields, isolate reference answers from all agent-visible context, expose actual dataset paths and benchmark selection, and connect S1's adapters to the shared task registry. Synthetic samples should remain explicitly synthetic.

Acceptance: real-format fixtures load meaningful prompts and separate evaluator references; rendered prompts contain no gold answers; each benchmark runs through the shared executor; raw task identity and release provenance remain traceable.

**S4-M4. Measure correctness and intelligibility independently of density.** Joint owner: S1 for protocol assessment.

Evidence: [trial result schema](/home/link4444/Downloads/Ai/project_blackboard/contracts/schemas.py:701), [simulated classification](/home/link4444/Downloads/Ai/project_blackboard/benchmarks/mscore.py:305), and [live classification](/home/link4444/Downloads/Ai/project_blackboard/benchmarks/mscore.py:491).

Results record agreement but no final answer, correctness score, or benchmark evaluator outcome. `ground_truth` passed to `run_single_trial` is unused. Strong and one-way intelligibility are hard-coded true; Ultra-Strong is assigned from density or intervention count. An ablation variable cannot also define the outcome it is supposed to explain.

Change: record the final prediction, official/task-appropriate correctness score, and consensus separately. Implement trace-based intelligibility assessment under the supervisor-approved PXP adaptation. Store tag counts in the defined assessment field rather than passing ignored extras.

Acceptance: unanimous wrong answers count as agreement but fail accuracy; low-density traces can achieve the same intelligibility level as equivalent high-density traces; disagreement/failure traces are not automatically Strong. The cited PXP implementation defines levels from directed tag sequences, which underscores the need to state the project's multi-agent adaptation. [PXP implementation paper](https://arxiv.org/html/2410.20600v1)

**S4-M5. Correct the imported executor path and record actual failures.**

Evidence: [executor invocation](/home/link4444/Downloads/Ai/project_blackboard/benchmarks/mscore.py:250), [main-only asyncio import](/home/link4444/Downloads/Ai/project_blackboard/benchmarks/mscore.py:521), and [live error handling](/home/link4444/Downloads/Ai/project_blackboard/benchmarks/mscore.py:454).

Using `BatchTrialRunner` as an imported library with an executor raises `NameError: asyncio is not defined`; the import exists only in the CLI block. Synchronous executors run directly inside an async function and can block the service. Live agent exceptions are skipped, and a run that simply exhausts its turn limit is labeled `DEADLOCK` even without a detected deadlock.

Change: import dependencies at module scope, run blocking inference away from the service loop, and record separate inference failure, timeout/budget exhaustion, unresolved deadlock, and successful outcomes. Retain attempted-call counts and failure reasons.

Acceptance: imported synchronous and asynchronous executors work; service endpoints stay responsive during inference; errors and budget exhaustion do not silently become ordinary deadlock observations.

**S4-M6. Make token accounting and broker lifetime trial-correct.** Joint owner: S2.

Evidence: [shared broker construction](/home/link4444/Downloads/Ai/project_blackboard/benchmarks/mscore.py:424), [per-trial context manager](/home/link4444/Downloads/Ai/project_blackboard/benchmarks/mscore.py:451), and [usage retrieval](/home/link4444/Downloads/Ai/project_blackboard/benchmarks/mscore.py:488).

One broker is captured for all trials, but lifetime usage is assigned to each result without a reset/difference. Two identical offline trials reported 1,126 and 2,252 tokens. The context manager also closes the broker's thread pool after the first trial, incompatible with subsequently using its asynchronous/pooled API. Sandbox and repair costs need inclusion under the same accounting rules.

Change: manage broker lifetime across the batch or instantiate per trial; derive isolated per-trial usage, including repairs and sandbox calls. Store input/output/live/sandbox components and distinguish unknown usage from zero.

Acceptance: equal scripted trials report equal cost regardless of execution order; a later trial can still use the pooled API; the batch total reconciles with all trial usage and failed calls to the extent provider usage is available.

**S4-M7. Connect storage/scheduler events to the streaming service and usable snapshots.** Joint owners: S1 and S3.

Evidence: [Redis publisher](/home/link4444/Downloads/Ai/project_blackboard/storage/redis_store.py:607), [FastAPI application](/home/link4444/Downloads/Ai/project_blackboard/streaming/main.py), [manager singleton](/home/link4444/Downloads/Ai/project_blackboard/streaming/manager.py:189), and [benchmark telemetry import](/home/link4444/Downloads/Ai/project_blackboard/benchmarks/mscore.py:34).

The store publishes to Redis channels, but the service has no Redis subscription bridge. The benchmark CLI imports a separate process-local manager, which does not update the singleton in an independently running uvicorn server. Store events also lack the full `contribution` or `snapshot` expected by the graph/history consumer. Healthy endpoints therefore do not establish a live connection to the reasoning system.

Change: provide one explicit event transport across processes, a subscriber bridge with a defined lifecycle, and canonical contribution/snapshot payloads. Hydrate current state on session attachment and support replay after disconnects. Send queue and lock events into the same session model.

Acceptance: run backend and trial process separately; a scheduler commit appears in the browser without manual HTTP posting; reconnecting reconstructs the real latest board and branch.

**S4-M8. Fix WebSocket cleanup, session isolation, and discovery.**

Evidence: [WebSocket hook](/home/link4444/Downloads/Ai/project_blackboard/frontend/src/hooks/useWebSocket.js) and [hard-coded URL](/home/link4444/Downloads/Ai/project_blackboard/frontend/src/App.jsx).

Effect cleanup closes the socket, but its `onclose` schedules another connection even during cleanup. This can revive old connections during session changes and React StrictMode remounting. Snapshot/deadlock/branch state is retained on session changes; global events can combine the latest snapshot from one session with a branch from another. The session list is populated only from the initial handshake. The `ws://hostname:8000` URL also bypasses the configured proxy and is unsuitable when the page uses HTTPS.

Change: cancel reconnects on disposal, invalidate old callbacks, reset or partition state by session, refresh discovery, bound/deduplicate event history, and derive an environment-configurable `ws`/`wss` endpoint.

Acceptance: switching sessions leaves exactly one intended connection and no cross-session state; unmount never reconnects; new sessions become selectable; configured secure/proxied deployment works. Browser testing is still required to verify the fix.

**S4-M9. Make demo status and benchmark actions truthful.**

Evidence: [demo fallback and batch callback](/home/link4444/Downloads/Ai/project_blackboard/frontend/src/App.jsx), [hard-coded dashboard](/home/link4444/Downloads/Ai/project_blackboard/frontend/src/components/AblationDashboard.jsx:4), and [default monitor values](/home/link4444/Downloads/Ai/project_blackboard/frontend/src/components/BottleneckMonitor.jsx).

The ablation dashboard always shows fixed sample percentages. Its Run button only calls a `console.log` callback and clears a timer; no job starts and no results load. A connected server with no events still shows the medical demo while the offline notice is hidden. The demo toggle cannot disable fallback when there are zero events. Lock status is never supplied, and an empty agent roster is displayed as three agents.

Change: expose an explicit demo mode with persistent visible labeling, load actual run summaries, and connect the button to a real job/status/results interface. Disable unavailable controls. Render absent telemetry as unknown/empty rather than measured values.

Acceptance: every displayed KPI traces to a selected run or is labeled demo data; the button creates a traceable job or clearly remains unavailable; a connected empty backend cannot masquerade as a live solved case.

**S4-M10. Visualize actual contribution relationships.**

Evidence: [graph viewer](/home/link4444/Downloads/Ai/project_blackboard/frontend/src/components/GraphViewer.jsx:66).

The viewer renders a horizontal sequence with arrows between adjacent turns. It never uses `target_contribution_id`, so it cannot show the branching relationships promised by the plan. The React Flow dependency is present but unused by the viewer. The requirement is the graph behavior, regardless of which library implements it.

Change: create edges from actual target relationships, show branch identity/fork points, and keep the chronological list as a supplementary view if useful.

Acceptance: when two contributions respond to one earlier node, the graph shows two edges from that node; a reply to a distant earlier claim is not shown as responding to its immediate predecessor.

**S4-M11. Correct misleading analytics classifications and denominators.**

Evidence: [recovery efficiency](/home/link4444/Downloads/Ai/project_blackboard/analytics/metrics.py:77) and [intelligibility plot](/home/link4444/Downloads/Ai/project_blackboard/analytics/plotting.py:138).

No encountered deadlock is reported as 100% recovery, even though recovery is undefined with a zero denominator. The intelligibility plot defines Strong as `100 - Ultra-Strong`, forcing NONE/ONE_WAY and failed classifications into Strong. Aggregation also pools benchmarks solely by density, which can hide changes in task composition.

Change: use explicit counts and N/A for unobserved recovery opportunities, plot measured mutually exclusive categories, and group by benchmark before presenting any pooled summary. Define whether recovery is per episode or per session.

Acceptance: zero-deadlock conditions display N/A, all intelligibility levels remain visible, and each rate exposes its numerator, denominator, and benchmark composition.

### Shared documentation and reproducibility

**TEAM-M1. Correct the research descriptions and unsupported empirical claims.** Lead: all students with the report author.

Evidence: [protocol reference](/home/link4444/Downloads/Ai/project_blackboard/references/baskar_pxp_protocol.md), [benchmark descriptions](/home/link4444/Downloads/Ai/project_blackboard/references/benchmarks_overview.md), and [failure-mode statistics](/home/link4444/Downloads/Ai/project_blackboard/references/multi_agent_failure_modes.md).

The reference notes describe MedAgentBench as diagnostic panel conferences, while its official workflow requires an interactive FHIR environment and external reference evaluation. KramaBench requires data-to-insight execution rather than merely discussing file names. The notes also present precise 85–90% late-error and 65% first-checkpoint recovery figures without a traceable study/table supporting them. The PXP note says four tags while listing a fifth `PROPOSE` entry and gives project-specific intelligibility definitions without identifying the adaptation.

Change: add accurate primary citations, distinguish official definitions from project adaptations, and remove or explicitly label unverified numbers as assumptions to test. Cite the exact dataset version. Describe current simulations and UI examples as demonstrations, and avoid treating generated clinical probabilities in the demo as measured findings. [Official MedAgentBench workflow](https://github.com/stanfordmlgroup/MedAgentBench), [Official KramaBench description](https://github.com/mitdbg/Kramabench)

Acceptance: each empirical claim is tied to a checkable source or the project's recorded experiment; the written methodology matches the executable system.

**TEAM-M2. Fix installable package coverage and runtime dependencies.** Lead: S1 and S2, with S4.

Evidence: [package configuration](/home/link4444/Downloads/Ai/project_blackboard/pyproject.toml).

Package discovery omits `prompts`, `streaming`, `benchmarks`, and `analytics`, although included modules import them. Persona JSON is not explicitly declared as package data, and project dependencies declare only Pydantic despite mandatory imports such as Redis in the storage module. Running from the source checkout hides these installation failures.

Change: include the intended runtime packages/assets and define core/optional dependency groups consistently with supported entry points. Validate an installed artifact outside the checkout.

Acceptance: a clean installed environment can import agents, load personas, run the chosen storage path, and invoke the documented benchmark/service entry points without relying on source-tree imports.

## Category 2: Recommended to change

### Student 1

1. **Add storage and cross-module tests that assert invariants.** Exercise a real Redis instance, lease-expiry during commit, concurrent status writes, rollback identity, store parity, and S2 agents inside S1 scheduling. Current concurrency tests use `InMemoryRedisClient` and do not establish real Redis transaction behavior. Keep the existing successful tests.
2. **Make scheduling repeatably deterministic and fair.** Use a monotonic insertion sequence for priority ties, deduplicate pending work, and define a starvation policy for persistent critics/priority agents. Timestamp ordering alone does not guarantee a reproducible tie under concurrent insertion. Record dispatch decisions for replay.
3. **Make storage mode and persistence expectations explicit.** Falling back to an in-memory store is reasonable for local development, but experiment startup should record the selected mode and avoid silently accepting an unintended mode. Define session retention, versioned status updates, and whether overwriting an existing session ID is permitted.

### Student 2

1. **Use a token/context and model-memory budget.** The semaphore caps calls, but it does not prevent several large models from overcommitting memory. `render_board` truncates text by characters and a fixed number of turns, which can omit the targeted assertion or needed evidence. Preserve the active hypothesis and target dependencies within a measured context budget.
2. **Retry selectively and validate settings.** Distinguish transient failures from missing models, invalid requests, and unsupported schema formats. Add bounded jitter/backoff where useful; validate temperature, generation limits, retries, panel size, and density inputs. Make optional LiteLLM installation/configuration match the documentation.
3. **Improve auditable inference instrumentation.** Record parse repairs, stale submissions, attempted calls, successful usage, missing usage, and failures by session and phase. Replace message-text matching for stale-turn detection with a typed exception contract. Extend the smoke script to export a concise validation summary and clean up the broker on early failure.

### Student 3

1. **Calibrate and consolidate scoring.** The simulator and final scorer use different objectives/thresholds. Evidence overlap, tag entropy, explanation length, and self-reported confidence are useful heuristics but need validation against judged correctness and actual recovery. Validate weight ranges/sums; state that lexical overlap is a lexical proxy. Do not call self-reported confidence calibrated without measurement.
2. **Bound and clean up resources per recovery.** Close sandbox sessions in success and failure paths, retain compact durable diagnostics, and cap prompt/token/time budgets as well as iteration counts. The daemon otherwise retains sessions and recovery objects until external cleanup.
3. **Add adversarial recovery examples.** Include a correct minority critic, fluent wrong agreement, malformed output, unavailable broker, evidence negation, multiple equally plausible branches, and early historical errors. Validate behavior using actual S2 schema output rather than only flat mock responses. A single RATIFY should not automatically establish a multi-agent consensus checkpoint.

### Student 4

1. **Add frontend and integrated streaming checks to CI.** Include `npm ci` and a production build, plus focused browser checks for session switching, StrictMode reconnect cleanup, real DAG edges, replay, and job progress. Current CI runs Python tests and schema export but does not build or exercise the frontend. Schema CI should compare generated artifacts, rather than merely exporting them.
2. **Report uncertainty and task-level comparisons.** After valid live data exists, include paired task/seed comparisons, confidence intervals, effect sizes, benchmark/domain breakdowns, and compute per correct resolved task. Use task-aware resampling so repeated runs of the same source task do not pretend to be independent tasks.
3. **Save a reproducible run manifest and checkpoints.** Record code commit, dataset release/hash, source task ID, run seed, actual capable-agent count, model assignment/quantization, decoding settings, hardware, backend, and execution mode. Save each result/failure incrementally so a long local-model batch can resume without rerunning completed work.
4. **Make playback coherent across the page.** The graph scrubs a contribution slice while the split panel and deadlock state can show the latest/full run. A new live turn forces the cursor to the end. Preserve paused historical inspection, rebuild all panels at the selected event, clear stale selected nodes, and expose readable empty/error states and keyboard-accessible controls.
5. **Strengthen boundary handling.** Add meaningful checks for empty datasets, missing/whitespace questions, missing labels, valid zero-valued answers, tiny turn budgets, requested trial counts not divisible by four, and result identity consistency. Truthiness-based answer fallback currently loses valid `0`/`False` values in several adapters.

### Shared

Provide one accurate setup and end-to-end demonstration guide. The main README's directory responsibilities and S4's test-count claim are stale. Document how storage, model serving, scheduler, solver, event bridge, and frontend start together. Include a small offline integrated demonstration and a separate real-model validation command. Ensure the plan/proposal version or stable references are available to teammates; broad `.gitignore` rules currently exclude PDFs and many local planning/review documents.

## Category 3: Change should be reviewed before applying

### Student 1

1. **Formal PXP adaptation.** Agree whether the project uses the exact pairwise finite-state model or a documented multi-agent simplification. Set the treatment of opening hypotheses, directional sender/receiver traces, explanation agreement, `REFUTE`/`REJECT` transitions, and recovery-created sessions. The source implementation defines Strong/Ultra-Strong from directed tag sequences; importing those names into a broadcast board requires an explicit aggregation rule. [PXP implementation paper](https://arxiv.org/html/2410.20600v1)
2. **Consensus quorum and deadlock thresholds.** Decide unanimity versus quorum, eligible ratifiers, and how useful dissent differs from a repeated impasse. Current components use different windows. Calibrate thresholds on a development set and freeze them before evaluation; do not simply increase thresholds to improve plots.
3. **Large concurrency/architecture changes.** Choose whether sequential deterministic commits with concurrent inference are sufficient, or whether a broader async scheduler/store redesign is warranted. Repair the demonstrated interface and atomicity bugs first; a wholesale rewrite needs a concrete benefit and migration plan.

### Student 2

1. **Homogeneous versus heterogeneous models.** Decide whether the primary experiment holds the base model constant and studies capability alone. A mixed-model panel is a valuable secondary study but introduces another variable. Actual model assignments must match the result manifest, rather than the runner's current placeholder names.
2. **Model, quantization, persona, temperature, and context changes.** These materially change disagreement rates and compute cost. Select and freeze the primary configuration, then treat alternatives as controlled experiments. Review any increase in parallel models against the team's available memory.

### Student 3

1. **Retrospective causal replay versus prospective mediation.** The plan promises alteration of a historical turn and re-scored continuation; current synthesis is a different operational method. Either implement the promised experiment or obtain approval to revise the research question. Agree on intervention granularity, baseline continuation, bounded replay depth, seed policy, outcome utility, and the strength of causal language justified by the simulator.
2. **Replacing the scoring objective.** An LLM judge, embeddings, factuality checks, or task correctness can improve evaluation, but each changes the objective and introduces cost/bias. Compare candidates on held-out development examples, record judge identity where used, and keep hidden labels out of branch selection. Avoid tuning on the final test outcomes.
3. **Backtracking depth and search strategy.** The three-checkpoint cap is a practical compute choice, not an established optimum. Review deeper search, beam search, or alternative rollback selection against measured recovery per token/time. The unsourced 85–90% figures should not decide the design.

### Student 4, jointly with Student 1

1. **Trial count and resampling design.** The proposal specifies 500 trials per configuration, rather than the CLI's 500 total. Across four densities and three benchmarks that is 6,000 runs. Decide whether to retain that target or approve a smaller statistically justified study. Use source task IDs and varying, paired role/model seeds for repetition. S1's MedAgentBench `role_seed` metadata is currently discarded when generating trial configs, so repeated IDs alone do not deliver the intended seed variation.
2. **Official environment evaluation versus a declared task subset.** KramaBench involves executing data pipelines over raw files; MedAgentBench requires its FHIR environment and reference evaluator. Implementing those tool/evaluation paths adds meaningful scope beyond parsing. Agree whether the final study uses the official harnesses or an explicitly named adaptation/subset, and restrict cross-benchmark claims accordingly. [KramaBench official repository](https://github.com/mitdbg/Kramabench), [MedAgentBench official repository](https://github.com/stanfordmlgroup/MedAgentBench)
3. **MSCoRe release and language policy.** The currently source-linked card differs from the plan's earlier description and is primarily Chinese with multiple task formats. Pin the intended release and subset, then decide whether to use native-language prompts or a reviewed translation protocol. Changing to a newer release or translating everything automatically changes the study population and should be documented. [MSCoRe dataset card](https://huggingface.co/datasets/032564yn/MSCoRe/blob/main/README.md)

## Suggested correction order and review acceptance

1. **First, protect validity:** label existing simulations, isolate evaluator answers, remove unsupported empirical claims, and define the protocol/experiment decisions that require supervisor review.
2. **Then repair shared integration:** canonical board contract, real-agent scheduler adapter, atomic storage, capability separation, and correct recovery dispatch.
3. **Complete the core reasoning path:** strict S2/S3 parsing, faithful bounded replay with a comparable baseline, candidate-isolated scoring, and verified live consensus after injection.
4. **Connect the UI:** actual cross-process events, complete snapshots/branches, clean session switching, and truthful job/results views.
5. **Validate a small study before scaling:** run a few real tasks from each approved benchmark path, verify correctness and cost accounting, then execute the agreed paired ablation and regenerate research figures.

For final review, each student should demonstrate the affected behavior and provide the corresponding trace/run artifact. The team should present one integrated example that starts from a real task, records a disagreement, explores an isolated alternative history, applies a valid recovery, reaches independently evaluated consensus, and displays the same events in the UI. Include a zero-capability control and a failed-recovery example.

No numerical student grades are assigned here. The current evidence supports an implementation review and a remediation plan; grading individual contribution would also require contribution history, milestone expectations, and each student's demonstration.

## Selected reproduced observations

These observations used the implementation's actual classes with offline model responses; they are additional review probes, not existing tests.

| Probe | Observed result |
| --- | --- |
| Register actual `PEXAgent` in scheduler | `TypeError: Agent handler for a is neither callable nor an agent instance.` |
| Wrap `agent.act` for scheduler using S2 board | `StaleTurnError: turn_index 1 != expected 0` |
| Second actual-agent turn on other in-memory board | `AttributeError: 'BlackboardEntry' object has no attribute 'target_contribution_id'` |
| Redis write with mismatched session, nonexistent RATIFY target, and index 99 | Accepted into history |
| Two RATIFY entries for `Answer A` and `Answer B` | Scheduler consensus predicate returned `True` |
| Controlled lock expiry between validation and save | Two successful commits; saved version 1; second writer's entry missing |
| MSCoRe snapshot rendered for agent | Gold `ground_truth` included in context |
| MSCoRe record with `instruction`/`input`/`output` | Empty question/answer; missing context |
| KramaBench-style `answer`/`answer_type` plus subtask answer | Reference fields lost; subtask gold answer exposed in agent context |
| Three-agent panel at 66% capability through round-robin | Only the non-capable critic selected, four consecutive dispatches |
| Inconsistent RATIFY claim | `Answer B` silently changed to `Answer A` |
| S2 nested JSON passed to sandbox sanitizer | Actual claim replaced by fallback |
| Invalid sandbox text | Defaults to `PXPTag.RATIFY` |
| Two identical live-executor trials with offline backend | Token costs 1,126 then 2,252 |
| Imported runner with executor | `NameError: name 'asyncio' is not defined` |
| Simulated zero-density condition, task `probe`, seed 1 | One counterfactual intervention recorded |
| Two-contribution Redis sandbox clone | Source session IDs retained; parent `first` dangling after IDs renamed |
| Recovery for existing agent | Original handler replaced |
| Contribution → flat entry → contribution | Token usage and latency become null; counterfactual flag becomes false |
