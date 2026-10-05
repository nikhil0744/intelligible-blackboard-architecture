# Student 3 implementation plan: historical replay, safe recovery, and measured attribution

Prepared: 5 October 2026.

Status: planning document only. The changes described below have not been implemented as part of creating this document.

## 1. What is already present

This plan is based on [compute-readiness.pdf](compute-readiness.pdf), particularly Student 3's assignment on pages 9–10 and the integration gates on pages 13–14, and inspection of the current repository.

**Student 1 and Student 2's work is present in the current directory and integrated into the checked-out branch, `s3-compute-readiness`, at commit `175f150`.** Separate worktrees for both students also exist.

| Area | Present in this checkout | Remaining limitation |
|---|---|---|
| Student 1 | Contracts, memory/Redis storage, scheduler, benchmark ingestion, tool-environment classes, tests | Canonical storage, atomic commits, position-based consensus, and historical environment checkpoints need further work. Existing environment implementations include simulated outputs. |
| Student 2 | `PEXAgent`, scheduler handler, stable personas and capability flags, validated parsing, shared broker, usage ledger, prompt budgeting, notebook setup and smoke tooling | Genuine model-driven tool interaction remains incomplete. Existing parsing validates ratification claims but needs the shared claim-and-explanation position contract. |
| Student 3 | Attribution analysis, sandbox manager, simulator, scorer, recovery daemon, tests | These implement an earlier approach and need substantial behavioural changes to meet the PDF. |

**Local verification during planning:** all **226 tests passed**. This validates the existing suite; several current tests still accept behaviours the PDF requires replacing. Live inference, a real Redis deployment, and official benchmark environments were not verified.

The selected scope is:

- Complete necessary shared Student 1/2 prerequisites yourself.
- Allow capable agents to revise **their own earlier contributions**.
- First deliver a verified recovery engine using real agent classes with explicitly scripted inference/tools. Official benchmark readiness is a later integration gate.

### Current recovery flow at a glance

The existing implementation is a **heuristic synthesis-and-reaction search**. When connected to the scheduler, its recovery path works approximately as follows:

```text
Scheduler detects a run of disagreement tags
    → Attribution engine follows parent links to a disputed contribution
    → Sandbox manager copies a window of conversation entries
    → Simulator generates a synthesized REVISE and one peer reaction
    → Heuristic scores decide whether to accept, retry, or backtrack
    → Solver returns a mediator-authored revision proposal
    → Scheduler queues the revision at priority zero and resumes
```

The components already provide useful infrastructure: contribution models, graph traversal, isolated conversation copies, model access, and a scheduler hook. However, generating a synthesis and receiving one favourable reaction does not demonstrate what would have happened under a different historical contribution. That requires the paired replay described below.

Throughout this document, **current implementation** means behaviour inspected at commit `175f150`; **proposed implementation** describes future work. Each implementation step now explains the current code, the reason for changing it, and the replacement. Source links are relative to this document so you can open the relevant files in VS Code.

## 2. What you are building—and why

Your job is to answer:

> If this agent had made a different contribution at that earlier point, would the subsequent discussion have produced a better outcome?

The implementation must establish three separate facts:

1. **Sandbox result:** an alternative history reached valid agreement.
2. **Live result:** applying its correction led the actual agents to agree.
3. **Evaluated result:** independent grading found the outcome better, worse, or unchanged.

Agreement alone does not establish correctness.

Use **paired historical replay**:

```text
Checkpoint immediately before historical turn k
    ├── Baseline: restore original turn k → regenerate continuation
    └── Intervention: original author replaces turn k → regenerate continuation
                                      ↓
                         Validate and select candidate
                                      ↓
                  Append one revision to the live history
                                      ↓
                       Obtain fresh live ratifications
```

| Approach | Why use it? | Improvement over the current implementation |
|---|---|---|
| Same checkpoint for baseline and intervention | Controls the information and environment available at the comparison point | Makes the comparison more informative than observing that an extra discussion turn helped |
| Same agents, parser, broker, and scheduler | Keeps ordinary and replay behaviour consistent | Removes the simulator's separate permissive response handling |
| Isolated tool-state forks | Makes alternative histories independent | Prevents one candidate's files or actions from affecting another |
| Three-agent consensus on one position | Gives recovery a concrete acceptance rule | Replaces confidence, explanation-length, and composite-score shortcuts |
| Hidden-answer-free selection | Preserves the integrity of the experiment | Removes answer-informed candidate ranking |
| Append-only live revision | Preserves what actually happened | Makes selected, applied, and confirmed recovery separately auditable |

This design supports **measured effects within the replay experiment**. It does not establish real-world causality or prove that an agent caused the original failure.

## 3. Implementation sequence

Implement these as small, dependency-ordered changes. Each step has an observable completion gate.

### Step 1 — Repair the shared protocol and storage foundation

This is required before replay results can be trusted.

**Current implementation**

- In [scheduler/engine.py](../../src/scheduler/engine.py), `DeterministicScheduler` defaults to `consensus_threshold=2` and `deadlock_window=2`. `_check_consensus()` checks trailing `RATIFY` tags from distinct agents; it does not establish that all three agents endorse the same claim-and-explanation position. `_check_deadlock()` checks disagreement tags and distinct agents without binding the disagreements to one active position.
- `PEXAgent` already generates a contribution without committing it, which is the correct separation to retain. Its [semantic validator](../../src/prompts/validation.py) rejects a `RATIFY` whose claim differs from its target, but there is no shared immutable claim-and-explanation position identity.
- [blackboard/store.py](../../src/blackboard/store.py) converts contributions to flat `BlackboardEntry` objects and appends them. This path does not perform the same version checks or version increments as the Redis path. The [conversion adapters](../../src/contracts/schemas.py) do not carry the dedicated `token_usage` and `latency_ms` fields through the flat representation.
- [storage/redis_store.py](../../src/storage/redis_store.py) checks the lease and version, then saves the modified snapshot through separate operations. The scheduler supplies a version read immediately before commit, rather than preserving the version used during generation. Some scheduler and store paths also use `len(history) + 1`, while agents generate zero-based indices.
- [scheduler/queue.py](../../src/scheduler/queue.py) breaks priority ties using timestamps and allows duplicate pending turns.

**Why change it?**

Suppose Agent A ratifies position X and Agent B ratifies position Y. The current scheduler's consensus check can accept the two trailing tags even though they concern different positions, while Agent C has not agreed at all. A replay scored using that check can therefore appear successful without meeting the study's consensus definition. Inconsistent storage and version handling also make it difficult to prove that a proposal was generated from the state it is being applied to.

**Proposed implementation — what you will change**

- Store canonical `AgentContribution` objects internally. Keep flat legacy entries behind explicit conversion adapters.
- Use zero-based turn indices everywhere; remove automatic reindexing that hides invalid submissions.
- Introduce an immutable **position identity** binding a claim to its explanation and evidence.
- Make `RATIFY` explicitly endorse that position. Changed claims or explanations require a different protocol action; validation must never rewrite disagreement into agreement.
- Require all three distinct roster agents to ratify the active position. Creating a proposal does not count as ratifying it.
- Trigger deadlock after three consecutive `REFUTE`/`REJECT` contributions concerning that position from at least two agents.
- Preserve the snapshot version used to generate a contribution and validate that version at commit.
- Make memory commits atomic under one lock. Use Redis optimistic transactions with retry, watching the session and lease keys so lease checks, version validation, history append, and status changes cannot race.
- Reject duplicate IDs, stale versions, wrong identities, invalid branch targets, expired leases, and terminal-session writes.
- Preserve instrumentation and deep-copy boundaries.
- Use stable insertion order for queue ties, deduplicate pending turns, and keep every persona eligible for ordinary turns.

**Completion gate:** the same three real `PEXAgent` instances, backed by scripted inference, complete equivalent multi-turn sessions against memory and Redis storage.

### Step 2 — Add historical checkpoints and actual tool interaction

A conversation copy is insufficient if tools have changed files or external state.

**Current implementation**

- In [sandbox/manager.py](../../src/sandbox/manager.py), `create_session_from_attribution()` copies entries within `start_step <= step_number <= rollback_step`. This retains the rollback contribution and can omit earlier context. It is a selected history window, not a complete checkpoint immediately before the contribution being replaced.
- [counterfactual/attribution.py](../../src/counterfactual/attribution.py) treats individual `RATIFY` entries or checkpoint metadata as consensus checkpoints. An individual ratification is not proof of three-agent consensus.
- The sandbox contains an in-memory board and simulated entries. It does not capture scheduler state or own a historical tool-environment checkpoint. Backtracking reloads conversation entries into the same sandbox object.
- [benchmarks/environment.py](../../src/benchmarks/environment.py) offers reset, action execution, fork, and close methods, but no explicit historical checkpoint/restore contract. The KramaBench fork copies only top-level files, and its copy destination differs from the workspace subsequently constructed for the child. Both environment forks shallow-copy observation models, so nested observation data is not guaranteed independent. Branch workspaces are created beneath the live workspace tree.
- Some tool responses are synthetic: KramaBench search constructs example records and correlation uses variable names and a seed; MedAgentBench supplies default clinical observations when data is absent. These are development fixtures, not evidence of official benchmark execution.
- [agents/base.py](../../src/agents/base.py) currently parses model output as a final agent decision; it has no action/observation loop connecting the agent to those tool environments. A tool timeout returns a failure observation, but the worker thread is not necessarily stopped.

**Why change it?**

If a file was created during turn 8, an agent reconsidering turn 3 must not be able to read it. Copying conversation messages without restoring the files and tool state can leak that future information. Similarly, restarting one candidate from another candidate's modified environment makes the candidates incomparable. A timed-out worker that keeps writing can break isolation after the caller thinks the action has finished.

**Proposed implementation — what you will change**

**Implement a checkpoint before every live agent turn**, capturing:

- The complete historical conversation prefix.
- Active position and ratification state at that point.
- Scheduler queue and policy state.
- Agent specifications and seed schedule.
- Environment files, mutable tool state, and observations available so far.

A checkpoint immediately before turn `k` must exclude turn `k` and everything after it. Capture it before that turn's tool actions begin.

Add these shared interfaces:

| Interface | Required responsibility |
|---|---|
| `ReplayCheckpoint` | Identify the historical boundary and reference board, scheduler, and environment snapshots |
| Environment `checkpoint`, `fork`, `restore`, `close` | Restore an exact historical state and provide independently disposable branches |
| Agent action-or-contribution response | Distinguish a tool request from a completed reasoning contribution |
| `ReplayResult` | Record outcome, validation, final position, usage, seed schedule, and trace references |

Extend Student 2's shared agent loop to validate tool requests, execute permitted actions, return observations to the model, and continue until it produces a valid contribution. Ordinary and replay turns must use this same loop.

Use independent workspaces outside the parent workspace tree. Copy nested files and mutable state correctly. Timed-out or cancelled tool workers must stop before a branch can be reused or closed.

**Completion gate:** changing a branch file, nested observation, or simulated database record cannot change the checkpoint, live environment, or sibling branch.

For this milestone, use explicit scripted environments. Existing synthetic outputs must be labelled as simulation; missing live data must produce an error.

### Step 3 — Replace the existing simulator with paired replay

**Current implementation**

- [counterfactual/solver.py](../../src/counterfactual/solver.py) and [counterfactual/simulator.py](../../src/counterfactual/simulator.py) construct `MockLLMBroker` automatically when no broker is supplied. A missing live dependency can therefore run through fixture behaviour.
- `run_adaptive_search()` selects a target and an opposing agent from the conflict participants, asks for a synthesis, and then asks for one peer reaction. It appends both entries to the current sandbox and may repeat the process, feeding the previous critique into the next attempt. There is no separately replayed unmodified baseline.
- `generate_synthesis_prompt()` includes the detected impasse, contested positions, and clashing evidence from the later conflict analysis. That information may not have existed at the historical turn being reconsidered.
- The simulator calls the broker directly using its own prompts. It does not run the normal three-agent scheduler or reconstruct the complete `PEXAgent` persona and decoding configuration for the historical author.
- `_extract_entry_fields()` and `_sanitize_entry_kwargs()` substitute default predictions, explanations, evidence, and confidence for missing or unusable fields. An absent or invalid peer tag becomes `RATIFY`. This bypasses the stricter validation and bounded failure behaviour already implemented by Student 2.
- The search does not use the panel's counterfactual-capability flag to restrict which author's earlier turns can be changed.

**Why change it?**

The current algorithm asks agents to resolve an already observed conflict, which is a different intervention from changing a contribution using only information available when it was originally made. For example, if turn 7 contains the decisive evidence, including it in a replacement for turn 2 gives that author hindsight. Also, without a baseline continuation, you cannot tell whether ordinary additional discussion would have reached the same result. Permissive defaults can make invalid model output look like successful reasoning.

**Proposed implementation — what you will change**

**Remove from the empirical recovery path:**

- Automatic mock-broker construction.
- Default successful responses or default `RATIFY`.
- Hidden-answer-based scoring.
- The separate permissive parser.
- Repeated synthesis inside an already modified sandbox.

Require an explicit `offline` or `live` mode. Offline mode requires an explicitly supplied scripted backend. Live mode requires a configured live backend; failures remain recorded failures.

**Candidate policy:**

- Only contributions authored by counterfactual-capable agents are eligible.
- Use the existing disagreement graph to order eligible contributions on the contested position's dependency path, newest first.
- Break ties by turn index and contribution ID.
- Examine at most three candidates.
- If none is eligible, record that outcome without inventing a replacement author.
- Treat structural attribution as a search diagnostic, never as proof of blame.

**For each candidate at turn `k`:**

1. Create two fresh forks from the checkpoint before `k`.
2. In the baseline, restore the original contribution and its recorded tool effects. Verify that the restored post-turn environment matches the recorded state.
3. In the intervention, ask the original author to reconsider using only the pre-turn information and a fixed, neutral reconsideration instruction.
4. Preserve persona, model, decoding settings, and ordinary tool access.
5. Replay up to six subsequent turns in each branch through the shared scheduler.
6. Disable recursive recovery inside both branches.
7. Persist both results and release their resources.

Use a recorded deterministic seed schedule based on trial seed, checkpoint, agent, and replay-turn index. Corresponding baseline/intervention continuation slots use matching seeds. Branch labels must not change ordinary prompt content.

Give both branches the same continuation limits and preallocated call/time allowances. If the shared trial budget prevents a complete comparison, record it as incomplete.

**Completion gate:** the replacement changes subsequent model inputs, while future contributions, future observations, and future active positions remain absent.

### Step 4 — Implement valid selection and measured attribution

Replace the current composite-score winner selection with the PDF's rule.

**Current implementation**

There are two heuristic scoring stages, serving different parts of the current search:

1. In [counterfactual/simulator.py](../../src/counterfactual/simulator.py), `compute_agreement_score()` weights the peer's PXP tag at `0.50`, evidence overlap at `0.25`, and explanation quality at `0.25`, with an overconfidence penalty. Explanation quality uses length and reasoning keywords. If `ground_truth` is present, a matching answer in the candidate text can increase that score. This stage chooses the best synthesis/reaction pair and controls search stopping.
2. In [counterfactual/scoring.py](../../src/counterfactual/scoring.py), `CreditAssignmentScorer.score_branch()` combines agreement, tag-entropy reduction, mean confidence, and a drift measure using default weights `0.50`, `0.25`, `0.15`, and `0.10`. The solver accepts a proposal when this branch score reaches its default threshold of `0.70` and a candidate exists; it does not require a complete three-agent replay consensus.

`assign_credit()` gives `REJECT` and `REFUTE` contributions preset positive blame scores, adjusted for evidence and confidence. Simulated revisions and ratifications receive favourable credit based on their tags and the branch score. The output calls these values `causality_score`, although no baseline-versus-intervention outcome comparison is performed.

**Why change it?**

A longer explanation or a confident agreement can raise a heuristic score without improving correctness. Answer-dependent scoring can select a different branch when only the hidden reference answer changes. Automatic blame for rejection can penalize the one agent correctly challenging a wrong majority. These measures can describe conversation structure, but they cannot establish the measured effect of a historical replacement.

**Proposed implementation — what you will change**

Use protocol validity and actual three-agent replay consensus as eligibility checks, then rank by measured cost. Keep structural and lexical measures only as explicitly labelled diagnostics. Compute outcome differences only after independent evaluation of a frozen baseline/intervention pair.

**A candidate is eligible only when:**

- Its contributions and environment actions pass validation.
- Its replay reaches three-agent consensus on one position.
- Its paired replay completed within the applicable limits.

Rank eligible candidates by:

1. Fewer measured intervention inference tokens, including replacement generation, repairs, tool-related model calls, and continuation.
2. Fewer regenerated continuation turns.
3. Stable candidate order.

Do not use confidence, explanation length, lexical overlap with an answer, or evaluator scores.

For a comparison containing incomplete token accounting, mark token cost unknown and skip the token criterion for that comparison; use turns and stable order. Record that limitation explicitly.

After selection is frozen, submit the selected intervention and its paired baseline to a separate evaluator boundary. Report:

- Baseline and intervention agreement.
- Baseline and intervention task scores, when an appropriate evaluator exists.
- Score difference: `intervention − baseline`.
- Tokens, calls, elapsed time, failures, and replay turns.

Keep baseline costs in total recovery cost even though candidate ranking uses intervention cost. If official evaluation is unavailable, record the score as unavailable.

**Completion gate:** changing evaluator-only answers cannot change prompts, tool observations, branch selection, or final predictions.

### Step 5 — Apply one live revision and confirm recovery

**Current implementation**

- [counterfactual/solver.py](../../src/counterfactual/solver.py) packages the winning synthesis as a `REVISE` authored by a configured mediator, by default `counterfactual_resolver`. It emits `COUNTERFACTUAL_RESOLVED` when it has generated the proposal, before fresh live ratification has occurred.
- The existing [ReviseProposal contract](../../src/contracts/schemas.py) carries the session, winning branch, target turn, contribution, and credit summary. It does not require the source live version, a proposal ID for duplicate rejection, or an explicit validation result.
- `inject_counterfactual_recovery()` in [scheduler/engine.py](../../src/scheduler/engine.py) creates a handler returning a copy of the proposed revision. It registers that handler as a new agent, or replaces the handler of an existing agent with it, and enqueues a priority-zero turn. It does not restore an existing agent's original handler afterward.
- The current path already appends a revision rather than replacing past history. That useful behaviour should be retained. However, proposal injection does not enforce source-version freshness, one-time application, or a separate three-agent live confirmation stage.

**Why change it?**

The live board may advance while a sandbox is running; applying a proposal against a different state can invalidate its assumptions. Replacing an ordinary handler can make future turns repeat the stored revision instead of generating new reasoning. A successful sandbox search also does not guarantee that the live agents will accept the correction, so reporting recovery when a proposal is merely generated overstates the result.

**Proposed implementation — what you will change**

Represent recovery as a single scheduler action, preserve the original author's identity and branch provenance, and keep the existing agent roster and handlers. Track selection, application, and live confirmation as separate states.

Extend the existing `ReviseProposal` contract with:

- Unique proposal and recovery-episode IDs.
- Source live-board version.
- Historical target and checkpoint identity.
- Selected branch provenance and validation result.
- One-time revision payload.

Update the scheduler to apply proposals as **one-time queue actions**. Its current implementation can replace an agent's ordinary handler; remove that behaviour.

At application:

- Atomically reject stale or previously applied proposals.
- Append a new `REVISE` to the live history.
- Preserve historical contributions unchanged.
- Create a new live position and clear prior ratifications for confirmation.
- Revalidate queued work and resume ordinary handlers.
- Do not copy sandbox ratifications or speculative tool mutations into live state.

Record success only after all three agents freshly ratify the live position. If they disagree, tools fail, or the budget expires, preserve that outcome even if the sandbox succeeded.

**Completion gate:** one proposal is applied once, ordinary agents continue normally, and a selected sandbox success can correctly result in failed live recovery.

### Step 6 — Enforce budgets, record events, and package the handoff

**Current implementation**

- `SolverConfig` in [counterfactual/solver.py](../../src/counterfactual/solver.py) defaults to five simulation iterations per search and up to three escalation levels. These are local search bounds, not the PDF's shared per-trial limits on live turns, recovery episodes, inference calls, and elapsed time.
- Student 2 already supplies a [usage ledger](../../src/llm_broker/ledger.py) with trial and phase filtering, unknown-usage handling, and optional incremental JSONL output. The simulator's request helper does not establish baseline/candidate usage scopes or carry its branch/checkpoint identity into those calls, so that accounting support is not yet fully connected to recovery.
- The solver keeps an in-memory recovery history and emits a success-oriented event. The shared event model has an event ID and timestamp, but no required ordered sequence or board-version field, and the recovery path does not emit the complete requested lifecycle.
- The solver exposes `cleanup()` and calls it when detached, but `resolve()` has no `finally` block guaranteeing cleanup on each return or exception. [SandboxManager](../../src/sandbox/manager.py) closes a session by removing it from its registry; its sessions do not own tool-environment resources to close or durable traces to flush.

**Why change it?**

Limiting search iterations does not bound the total inference workload once repairs, retries, baseline replays, and repeated recoveries are included. Missing phase metadata can obscure the real cost of recovery. A failure before cleanup can leave resources behind, and an in-memory result can disappear when a notebook runtime stops. Ordered lifecycle records let Student 4 distinguish a selected candidate from a confirmed live recovery.

**Proposed implementation — what you will change**

Reuse the existing ledger and add recovery attribution to it. Enforce a shared trial budget, persist complete lifecycle traces, and make every branch responsible for closing its environment after its trace has been saved.

Use the PDF's limits:

| Limit | Default |
|---|---:|
| Historical candidates per episode | 3 |
| Regenerated continuation turns per branch | 6 |
| Recovery episodes per trial | 2 |
| Total live turns, including applied revisions | 15 |
| Shared inference-call ceiling | 100 |
| Shared trial duration | 20 minutes |

The intervention replacement is additional to its six continuation turns, and all its calls count toward the shared ceiling. Count backend attempts and repair calls; retries must not bypass the budget.

Reuse Student 2's shared broker and ledger. Add branch, checkpoint, and episode identifiers to call records. Keep baseline, candidate, ordinary, and evaluator usage distinguishable; evaluator costs belong to a separate grading ledger.

Emit ordered lifecycle events for trigger, branch creation, evaluation, selection, application, confirmation, failure, and exhaustion. Include stable event IDs, sequence numbers, session IDs, and board versions. Keep evaluator answers out of ordinary events.

Persist traces before cleanup. Use `finally` paths for success, failure, cancellation, and exceptions, including partial branch construction.

Deliver:

- A replay-and-attribution methodology guide explaining the experiment and its limits.
- An offline integration command producing ordinary-consensus, successful-recovery, and failed-recovery traces.
- A live smoke command reusing Student 2's inference settings.
- Versioned recovery artifacts and exported schemas.
- Student 4 integration instructions covering invocation, events, outcomes, and artifact consumption.

## 4. Tests and acceptance criteria

Extend the suite around behaviours that could invalidate the study.

| Test | Required outcome |
|---|---|
| Historical replacement | Later replay prompts contain the replacement, not the original turn |
| Future-information exclusion | Later facts, observations, and active positions never reach the earlier agent |
| Answer isolation | Changing hidden labels leaves inference and selection unchanged |
| Baseline equivalence | Both branches start from equivalent historical board and tool state |
| Three-agent consensus | Two ratifications, duplicate voters, or conflicting positions cannot succeed |
| Correct minority criticism | Criticism is not automatically assigned negative credit |
| Invalid output | Bounded repairs end in recorded failure, never fabricated agreement |
| Capability conditions | Zero-capability trials create no recovery branches; other conditions revise only eligible authors |
| Stale or duplicate proposal | Application is rejected without rewriting history |
| Handler preservation | Ordinary agent behaviour survives recovery application |
| Environment isolation | Nested files, observations, and mutable records cannot leak across branches |
| Cancellation and timeout | Work stops, traces persist, and resources close |
| Usage reconciliation | Trial totals include ordinary, baseline, intervention, repair, and failed calls |
| Real Redis concurrency | Lease expiry and simultaneous status/contribution updates cannot lose state |

Retain valid existing tests and rewrite those that explicitly encode the superseded behaviour. Run the full suite, schema-export comparison, Python 3.11/3.12 checks, and real Redis integration tests.

The first milestone is complete when the offline integrated traces demonstrate correct ordinary deliberation, paired replay, one-time application, successful live confirmation, and truthful failure handling.

## 5. Defaults and boundaries

- Reuse the existing Python architecture and public imports where practical.
- Keep the PDF's shared Qwen model, three stable personas, temperature `0.3`, context limit `8192`, response limit `768`, two repairs, and concurrency one.
- Use the same ordinary scheduler policy across live runs, replay, and capability conditions.
- Preserve dataset language.
- Describe seeded runs as controlled and auditable; do not promise bit-identical GPU outputs.
- Keep historical simulation artifacts readable and explicitly classified as legacy.
- Student 3's first milestone does not include the full experiment runner, dashboard, or deployment of official benchmark environments.
- Before reporting official benchmark performance, separately verify real environment execution, historical checkpoint support, official evaluation, and a real-model smoke run.

**Start with the shared protocol and checkpoint foundation.** Candidate generation becomes useful only after the system can preserve historical information, validate genuine agreement, and apply a correction safely.
