# Historical plans and implementation status

**Inactive archive:** these documents are retained for historical context. None is an active roadmap, assignment, or instruction to implement changes. See the [current project review](../../PROJECT_STATUS.md) for the assessed state of the code.

Reviewed against the local `prototype-live-demo` implementation on 12 November 2026. This is an inventory of existing documents, not a replacement roadmap or authorization to implement them. Historical checkboxes, completion percentages, ownership, and claims inside these documents are preserved; they are not the current assessment.

## Folder convention

- `fully_implemented/`: whole plans whose stated behavior and completion criteria are satisfied. Currently empty; no reviewed whole document qualifies.
- `not_started/`: proposed work with no implementation of its specific changes found. Existing supporting infrastructure does not count as implementation of a new corrective plan.
- Files directly in this folder: partially implemented plans, reviews containing mixed recommendations, and mixed guides/presentation material containing unfinished commitments.

Status describes the current code, not when work began or who performed it. An absent implementation is not proof that nobody investigated or attempted it. No new model runs or official benchmark evaluations were performed for this classification.

## Not started

| Document | Original location | Basis |
| --- | --- | --- |
| [Student 3 corrective plan](not_started/STUDENT3_PLAN.md) | `docs/guides/STUDENT3_PLAN.md` | Its six-phase corrective implementation has not been found: implicit mocks and default RATIFY remain, nested output is still discarded, there is no paired historical baseline/intervention replay, and recovery lacks version-safe application and confirmed-live lifecycle handling. Existing attribution, sandbox, and synthesis modules predate these requested corrections. |

## Started but not fully implemented

| Document | Original location | Present | Missing or unresolved |
| --- | --- | --- | --- |
| [Original four-person project plan](Blackboard_Agents_Project_Plan.pdf) | `docs/specs/Blackboard_Agents_Project_Plan.pdf` | Contracts, storage, scheduling, agents, recovery scaffolding, ingestion, streaming, dashboard, analytics, and tests. | Shared real-agent integration, protocol enforcement, faithful historical replay, connected telemetry, independently evaluated multi-benchmark study. |
| [Project overview and timeline](PROJECT_OVERVIEW.md) | `docs/guides/PROJECT_OVERVIEW.md` | Most named subsystems exist. | The promised integrated execution and research/evaluation milestones are not complete. |
| [Task 4 scoring plan and guide](task4_scoring_plan_and_guide.md) | `docs/guides/task4_scoring_plan_and_guide.md` | `BranchScoreDetails`, `PlateauDetector`, `CreditAssignmentScorer`, entropy, drift, composite scores, credit deltas, and contract construction exist in `src/counterfactual/scoring.py`. | The guide's contract example differs from the current schema. Its single-turn fallback/flag and substantive causal-reasoning defense against keyword parroting are absent. Credit remains heuristic, not established causal attribution. Core scoring is implemented, but the entire guide is not satisfied. |
| [Cross-branch review and integration plan](CROSS_BRANCH_REVIEW_REPORT.md) | `docs/reviews/CROSS_BRANCH_REVIEW_REPORT.md` | Agent, UI, scheduler, and ingestion modules named as missing in this historical review now exist; frontend consumes some counterfactual events. | Cross-process event delivery, valid live trial execution, and empirical ablation completion remain unresolved. Historical merge verdicts are not current branch instructions. |
| [Professor review and correction recommendations](PROFESSOR_REVIEW.md) | `docs/PROFESSOR_REVIEW.md` | Packaging/source layout, example labeling, and temporary schema comparison in CI address some recommendations. | Most agent/store integration, atomicity, replay, evaluation, and UI issues documented in the review remain. This is a review with mixed recommendations, not a single implementation task. |
| [Compute-readiness team plan](compute-readiness.pdf) | `docs/guides/compute-readiness.pdf` | Colab Ollama setup, a real-question smoke command, and temporary schema comparison in CI cover portions of the requested preparation. | Canonical integrated deliberation, official tool environments/evaluators, historical replay, per-call durable accounting, resumable experiments, truthful live artifacts, pilot, and verification gates remain incomplete. The plan explicitly treats earlier packaging work as its starting point, not a new deliverable. |
| [Student 2 historical guide and checklist](student2.md) | `docs/guides/student2.md` | Model broker, personas, constrained output, parsing/repair, local agent lifecycle, and parallel calls. | Its integration handoff is unfinished: the scheduler does not directly accept `PEXAgent`, turn conventions disagree, and the alternative board stores incompatible flat entries. Local-model validation checklist completion is not established by this archive. |
| [Presentation with next-sprint commitments](PRESENTATION_SLIDES.md) | `PRESENTATION_SLIDES.md` | Existing subsystem/demo descriptions accompany the proposed next sprint. | The listed Qwen/DeepSeek/Claude pairing, semantic embedding specialty routing, and SWE-bench expansion have no corresponding implementations. Generic configurable model support and a KramaBench adapter do exist, so the whole mixed document is not classified as wholly unstarted. Presentation claims are historical, not verification evidence. |

## Evidence and scope

Classification uses `src/scheduler/engine.py`, `src/agents/`, `src/blackboard/store.py`, `src/storage/redis_store.py`, `src/counterfactual/`, `src/sandbox/manager.py`, `src/benchmarks/mscore.py`, `src/streaming/`, the frontend, notebook, and CI workflow. The earlier repository analysis reproduced integration failures and verified 179 passing component tests; passing those tests does not satisfy missing plan acceptance criteria.

Documents were moved intact, except for relative reference links in the cross-branch review. Some historical documents already contain stale source paths, claims, and test counts; those were not rewritten during this organization task.

Setup/reference documents without an implementation plan, source comments, generated output copies, assistant role templates, and presentation binaries remain at their existing locations. The Student 4 guide describes existing components and setup rather than an unfinished work checklist. Git history and other branches were not reorganized.

Several collected documents were previously ignored local files. The scoped `.gitignore` exception makes this folder eligible for version control; nothing has been staged, committed, or pushed.
