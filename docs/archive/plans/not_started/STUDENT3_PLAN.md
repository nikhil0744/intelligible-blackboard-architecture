# Student 3 — Counterfactual Recovery: Responsibilities and Implementation Plan

Status: proposed implementation plan for review. Creating this document does not implement the changes below.

## 1. What your work is about

You own the system that investigates a deadlock by asking:

> What if an agent had made a different assertion earlier? Would replaying the conversation from that point produce a better outcome?

Your system must preserve the live run, create isolated alternative histories, compare them with an unchanged baseline, and propose a correction only when an alternative passes validation.

The main flow is:

```text
Live disagreement
        ↓
Identify an earlier contribution to investigate
        ↓
Checkpoint conversation and tool state
        ↓
Fork unchanged baseline and intervention branches
        ↓
Replace one historical contribution in the intervention
        ↓
Replay both through the original agents and scheduler
        ↓
Validate and select a candidate without reference answers
        ↓
Submit a one-time revision to the live scheduler
        ↓
Ordinary agents respond and confirm—or reject—recovery
```

A successful simulated branch is **not yet successful live recovery**. The live agents must subsequently ratify the same claim and explanation.

## 2. Your ownership and dependencies

You primarily own:

- `src/counterfactual/`: attribution, replay, candidate evaluation, recovery proposals.
- `src/sandbox/`: isolated histories, environment forks, resource lifecycle.
- Corresponding recovery, attribution, scoring, and sandbox tests.

You also own the documentation explaining what was changed, how replay works, and what the measured attribution means.

| Teammate | What you need from them | What you provide |
|---|---|---|
| Student 1 | Canonical snapshots, checkpoints, environment isolation, consensus validation, version-safe one-time recovery | Validated proposal and branch provenance |
| Student 2 | Real agent roster/configuration, strict parser, replay-compatible inference, per-call accounting | Replay requests and recorded recovery failures |
| Student 4 | Trace persistence, evaluator interface, event delivery, result aggregation | Branch trajectories, lifecycle events, baseline/intervention comparisons |

**Boundary:** you do not independently implement benchmark graders, model serving, the dashboard, or the authoritative scheduler.

Work against explicit adapters while teammate interfaces are being completed. An unavailable required environment must produce a recorded failure—not a fabricated substitute.

## 3. Problems in the current implementation

These are the specific behaviors the implementation will correct:

| Current behavior | Required behavior |
|---|---|
| Generates a new compromise and one peer response | Changes a historical contribution and replays a bounded continuation |
| May substitute mock inference when no broker exists | Mock mode must be explicitly selected |
| Invalid responses can become default RATIFY output | Invalid responses remain failures after bounded repairs |
| Separate parsing can lose Student 2’s nested payload | Reuse the validated agent parser and canonical contribution |
| Search candidates accumulate in one sandbox | Each candidate starts from its own isolated checkpoint |
| Branch selection can read a reference answer | Reference answers remain evaluator-only |
| Credit is assigned using fixed tag penalties | Report measured baseline/intervention differences |
| Proposal creation emits “resolved” | Resolution requires confirmed live consensus |
| Cleanup depends on external calls | Every completion, error, and cancellation closes its resources |

Structural divergence tracing remains useful, but it is a **diagnostic heuristic**, not proof that an agent caused an incorrect result.

## 4. Implementation order

### Phase 1 — Strict inputs and explicit failures

**Corresponds to S3.1.**

Remove implicit mock fallbacks and fabricated successful responses.

Use Student 2’s parser so prediction, explanation, confidence, evidence, and target identity survive intact. Preserve failed raw responses and repair reasons through the inference trace.

Distinguish missing dependencies, malformed output, inference failure, and exhausted budgets.

**Acceptance:** a valid nested contribution survives unchanged; malformed output cannot increase agreement or create recovery.

### Phase 2 — Faithful checkpoints and isolated forks

**Foundation for S3.2 and S3.3.**

Capture the conversation prefix, task facts, active position, agent configuration, scheduler state, and tool-environment checkpoint immediately before the investigated contribution.

Rebuild leading state from that prefix. Preserve valid graph relationships and source identities through explicit mappings.

Each baseline or candidate gets its own conversation and environment fork. Failed candidates cannot modify the live run or influence another candidate.

**Acceptance:** the checkpoint contains no future contributions or observations; sandbox tool writes are isolated; the live snapshot remains unchanged.

### Phase 3 — Historical intervention and replay

**Corresponds to S3.2 and S3.3.**

Use the original agent/persona and shared model interface to generate a replacement for the selected historical contribution. Mark the intervention explicitly.

Replay subsequent turns through the shared scheduler and original roster. Preserve model settings and a recorded seed schedule.

At each investigated checkpoint, run:

- An unchanged baseline continuation.
- A continuation containing the historical intervention.

Use matching continuation budgets. Do not provide either branch with later live-run history or hidden evaluator information.

Start with at most three checkpoint candidates and six continuation turns per branch.

**Acceptance:** replacing the historical contribution changes subsequent replay inputs; the baseline and candidate otherwise start from equivalent state.

### Phase 4 — Candidate validation and measured attribution

**Corresponds to S3.4.**

Replace competing heuristic success thresholds with the shared selection rule:

1. Contributions and tool actions must be valid.
2. All three agents must ratify the same position.
3. Among eligible candidates, prefer fewer measured inference tokens, then fewer replay turns, then stable candidate order.

Record unknown token usage explicitly; do not claim a measured token advantage where usage is unavailable.

Keep entropy, lexical overlap, confidence, and structural tracing as optional diagnostics. They cannot substitute for consensus or use reference answers.

After selection, allow Student 4’s evaluator to compare baseline and intervention outcomes. Report consensus differences and available correctness-score differences separately. If grading is unavailable, correctness attribution remains unavailable.

**Acceptance:** changing only the hidden answer cannot change prompts, predictions, or candidate selection. Fluent wrong agreement can pass agreement validation but must fail independent correctness evaluation.

### Phase 5 — Safe application and honest lifecycle events

**Corresponds to S3.5.**

Build a recovery proposal containing the selected correction, source live version, intervention identity, branch identity, and validation evidence.

Ask Student 1’s scheduler to apply it as a one-time revision. Preserve the main history and normal agent handlers.

Reject stale proposals rather than applying them against a changed board. Resume ordinary deliberation and require fresh live ratification.

Sandbox tool mutations are never copied directly into the live environment. Any live actions go through the authoritative tool interface.

Emit distinct events for:

- Recovery triggered.
- Branch created and evaluated.
- Candidate selected.
- Revision applied or rejected.
- Live recovery confirmed or failed.
- Budget exhausted.

**Acceptance:** creating or applying a proposal does not itself clear the deadlock. Only the scheduler’s confirmed live outcome does.

### Phase 6 — Budgets, cleanup, and handoff

**Corresponds to S3.6.**

Use the agreed initial defaults:

| Limit | Default |
|---|---:|
| Recovery episodes per trial | 2 |
| Checkpoint candidates per episode | 3 |
| Continuation turns per branch | 6 |
| Total live turns, including applied revisions | 15 |
| Total inference calls across the trial | 100 |
| Trial wall-clock ceiling | 20 minutes |

Recovery calls, baseline calls, repairs, and retries consume the shared trial budget. Do not reset that budget when creating a new branch.

Persist traces before closing sandboxes. Cleanup runs after success, failure, timeout, cancellation, and exceptions.

Provide one offline integrated demonstration and machine-readable artifacts for Student 4. Real-model validation follows after shared integration passes.

## 5. Tests and completion criteria

### Required regression tests

- Valid Student 2 output retains its complete nested payload.
- Missing live broker never silently activates mocks.
- Invalid JSON or tags cannot become successful ratification.
- Checkpoints exclude future assertions and tool observations.
- Failed candidates cannot change another candidate’s starting state.
- Baseline and intervention preserve matching configurations and budgets.
- Hidden-label changes cannot affect inference or selection.
- A correct minority critic is not automatically assigned causal blame.
- Selected trajectory, recorded score, and proposed revision refer to the same candidate.
- Stale proposals are rejected; recovery is applied at most once.
- An applied revision followed by dissent remains unresolved.
- All resources close after cancellation or failure.
- Zero-capability trials execute no recovery calls.

### Your work is complete when

An integrated trace shows a genuine historical intervention, an unchanged baseline, isolated replay, label-free candidate selection, safe live application, and subsequent confirmation or rejection.

A second trace demonstrates failed recovery. Both include measured usage, failure reasons, branch identities, and cleanup evidence.

Describe results as measured effects **within the replay experiment**. Do not claim proven real-world causality, correctness from agreement alone, or formal Strong/Ultra-Strong intelligibility from capability density.

The implementation will proceed in the phase order above, using small changes and focused regression tests before broad integration.
