"""Data contracts and schemas for the Intelligible Blackboard Architecture.

This module defines Pydantic v2 schemas for the multi-agent blackboard system,
grounded in the PXP (Prediction-eXplanation Protocol) by Baskar et al. (2025)
and Donald Michie's framework for machine intelligibility.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _utc_now() -> datetime:
    """Return the current time in UTC with timezone awareness."""
    return datetime.now(timezone.utc)


def _generate_id() -> str:
    """Generate a unique UUID string."""
    return str(uuid4())


# ============================================================================
# Protocol Tags & Enumerations
# ============================================================================

class PXPTag(str, Enum):
    """PXP (Prediction-eXplanation Protocol) interaction tags (Baskar et al., 2025).

    Every contribution submitted by an agent to the blackboard must be tagged
    with one of these four operational labels:
      - RATIFY: The agent agrees with both the prediction and the explanation.
      - REVISE: The agent offers an alternative prediction and/or explanation,
                updating the shared understanding or its internal model.
      - REFUTE: The agent disputes the prediction or explanation with
                counter-evidence or critique without proposing an adopted revision.
      - REJECT: The agent fundamentally rejects both the prediction and the
                explanation, signaling impasse or irreconcilable inconsistency.
    """
    RATIFY = "RATIFY"
    REVISE = "REVISE"
    REFUTE = "REFUTE"
    REJECT = "REJECT"

    @classmethod
    def _missing_(cls, value: object) -> Any:
        if isinstance(value, str) and value.upper() == "PROPOSE":
            return cls.REVISE
        return None


PXPTag.PROPOSE = PXPTag.REVISE  # type: ignore[attr-defined]


class BlackboardStatus(str, Enum):
    """Operational status of a blackboard reasoning session."""
    INITIALIZING = "INITIALIZING"
    ACTIVE = "ACTIVE"
    CONSENSUS = "CONSENSUS"
    DEADLOCK = "DEADLOCK"
    RESOLVED = "RESOLVED"
    TERMINATED = "TERMINATED"


class IntelligibilityLevel(str, Enum):
    """Michie-Baskar intelligibility criteria tiers.

    - NONE: No comprehensible explanation or mutual agreement.
    - ONE_WAY: The human / evaluator understands the agent's explanation.
    - STRONG: Every interaction cycle is one-way intelligible to the participants.
    - ULTRA_STRONG: System is strongly intelligible AND knowledge transfer
                    occurs (improving agent or human performance via feedback).
    """
    NONE = "NONE"
    ONE_WAY = "ONE_WAY"
    STRONG = "STRONG"
    ULTRA_STRONG = "ULTRA_STRONG"

    @classmethod
    def _missing_(cls, value: object) -> Any:
        if isinstance(value, str) and value.upper() == "WEAK":
            return cls.ONE_WAY
        return None


IntelligibilityLevel.WEAK = IntelligibilityLevel.ONE_WAY  # type: ignore[attr-defined]


class AgentRole(str, Enum):
    """Domain and operational roles for blackboard agents."""
    PRIMARY = "PRIMARY"
    CRITIC = "CRITIC"
    DOMAIN_EXPERT = "DOMAIN_EXPERT"
    ARBITER = "ARBITER"
    COUNTERFACTUAL = "COUNTERFACTUAL"


class BenchmarkType(str, Enum):
    """Supported multi-agent benchmark datasets."""
    KRAMABENCH = "KramaBench"
    MSCORE = "MSCoRe"
    MEDAGENTBENCH = "MedAgentBench"


class TelemetryEventType(str, Enum):
    """Event types broadcasted via WebSocket to the visualization UI."""
    BOARD_INITIALIZED = "BOARD_INITIALIZED"
    STATE_MUTATION = "STATE_MUTATION"
    AGENT_QUEUED = "AGENT_QUEUED"
    TURN_DISPATCHED = "TURN_DISPATCHED"
    AGENT_SUBMISSION = "AGENT_SUBMISSION"
    LOCK_ACQUIRED = "LOCK_ACQUIRED"
    LOCK_RELEASED = "LOCK_RELEASED"
    DEADLOCK_DETECTED = "DEADLOCK_DETECTED"
    COUNTERFACTUAL_TRIGGERED = "COUNTERFACTUAL_TRIGGERED"
    COUNTERFACTUAL_BRANCH_EVALUATED = "COUNTERFACTUAL_BRANCH_EVALUATED"
    COUNTERFACTUAL_RESOLVED = "COUNTERFACTUAL_RESOLVED"
    CONSENSUS_REACHED = "CONSENSUS_REACHED"


# ============================================================================
# Prediction & Explanation (PEX) Models
# ============================================================================

class Prediction(BaseModel):
    """Structured 'What' component of an agent's assertion."""
    model_config = ConfigDict(extra="ignore")

    claim: str = Field(..., description="The predicted claim, diagnosis, answer, or hypothesis.")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score in the prediction range [0.0, 1.0]."
    )
    probabilities: Optional[Dict[str, float]] = Field(
        default=None,
        description="Class or outcome probability distribution, if available."
    )
    summary: Optional[str] = Field(
        default=None,
        description="Brief natural language summary of the claim."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Domain-specific structured prediction fields."
    )


class Explanation(BaseModel):
    """Structured 'Why' component of an agent's assertion."""
    model_config = ConfigDict(extra="ignore")

    rationale: str = Field(..., description="Detailed reasoning steps or chain of thought.")
    evidence: List[str] = Field(
        default_factory=list,
        description="Supporting observations, citations, premises, or clinical findings."
    )
    assumptions: List[str] = Field(
        default_factory=list,
        description="Assumptions or constraints under which the explanation holds."
    )
    limitations: Optional[str] = Field(
        default=None,
        description="Known uncertainties, edge cases, or counter-considerations."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Domain-specific explanation metadata."
    )


class PEXPayload(BaseModel):
    """Coupled Prediction + Explanation (PEX) payload conforming to PXP standards."""
    model_config = ConfigDict(extra="ignore")

    prediction: Prediction = Field(..., description="The prediction ('what').")
    explanation: Explanation = Field(..., description="The explanation ('why').")


# ============================================================================
# Agent Contribution & PXP Turn
# ============================================================================

class AgentContribution(BaseModel):
    """A single atomic contribution posted by an agent to the shared blackboard."""
    model_config = ConfigDict(extra="ignore")

    contribution_id: str = Field(
        default_factory=_generate_id,
        description="Unique identifier for this contribution."
    )
    session_id: str = Field(..., description="Blackboard session ID.")
    turn_index: int = Field(..., ge=0, description="Sequential turn index.")
    agent_id: str = Field(..., description="Identifier of the contributing agent.")
    agent_role: Optional[AgentRole] = Field(
        default=None,
        description="Role played by the contributing agent."
    )
    tag: PXPTag = Field(..., description="PXP operational tag (RATIFY/REVISE/REFUTE/REJECT).")
    target_contribution_id: Optional[str] = Field(
        default=None,
        description="ID of the prior contribution being ratified, revised, refuted, or rejected."
    )
    payload: PEXPayload = Field(
        ...,
        description="The PEX (Prediction + Explanation) content of this turn."
    )
    is_counterfactual: bool = Field(
        default=False,
        description="Flag indicating if contribution originated in retrospective sandbox simulation."
    )
    token_usage: Optional[int] = Field(
        default=None,
        ge=0,
        description="Inference tokens consumed formulating this turn."
    )
    latency_ms: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Execution latency in milliseconds."
    )
    timestamp: datetime = Field(
        default_factory=_utc_now,
        description="UTC timestamp of the submission."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context or instrumentation telemetry."
    )

    @property
    def entry_id(self) -> str:
        return self.contribution_id

    @property
    def parent_id(self) -> Optional[str]:
        return self.target_contribution_id

    @parent_id.setter
    def parent_id(self, value: Optional[str]):
        self.target_contribution_id = value

    @property
    def step_number(self) -> int:
        return self.turn_index

    @step_number.setter
    def step_number(self, value: int):
        self.turn_index = value

    @property
    def prediction(self) -> str:
        return self.payload.prediction.claim if self.payload and self.payload.prediction else ""

    @property
    def explanation(self) -> str:
        return self.payload.explanation.rationale if self.payload and self.payload.explanation else ""

    @property
    def evidence_refs(self) -> List[str]:
        if self.payload and self.payload.explanation:
            return self.payload.explanation.evidence or []
        return []

    @property
    def confidence(self) -> float:
        return self.payload.prediction.confidence if self.payload and self.payload.prediction else 0.0

    @property
    def branch_id(self) -> str:
        return self.metadata.get("branch_id", "main")

    @branch_id.setter
    def branch_id(self, value: str):
        self.metadata["branch_id"] = value


# ============================================================================
# Blackboard State & History
# ============================================================================

class BlackboardSnapshot(BaseModel):
    """Complete snapshot of the blackboard state at a given point in time."""
    model_config = ConfigDict(extra="ignore")

    session_id: str = Field(..., description="Unique session identifier.")
    task_id: str = Field(..., description="Benchmark or task identifier.")
    task_description: str = Field(..., description="Task prompt or problem statement.")
    initial_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Background data, clinical records, or question context."
    )
    status: BlackboardStatus = Field(
        default=BlackboardStatus.INITIALIZING,
        description="Current operational status of the board."
    )
    version: int = Field(
        default=0,
        ge=0,
        description="Monotonic state version counter for optimistic locking."
    )
    active_claim: Optional[Prediction] = Field(
        default=None,
        description="The currently leading or ratified consensus prediction."
    )
    contributions: List[AgentContribution] = Field(
        default_factory=list,
        description="Immutable chronological sequence of all posted contributions."
    )
    active_agents: List[str] = Field(
        default_factory=list,
        description="List of agent IDs registered in this reasoning session."
    )
    created_at: datetime = Field(
        default_factory=_utc_now,
        description="Session creation UTC timestamp."
    )
    updated_at: datetime = Field(
        default_factory=_utc_now,
        description="Last state mutation UTC timestamp."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom session attributes, benchmark specs, or seeds."
    )

    @property
    def entries(self) -> List[AgentContribution]:
        return self.contributions

    @entries.setter
    def entries(self, value: List[AgentContribution]):
        self.contributions = value

    @property
    def active_branches(self) -> List[str]:
        if "active_branches" in self.metadata:
            return self.metadata["active_branches"]
        b = set()
        for c in self.contributions:
            b.add(c.metadata.get("branch_id", "main"))
        return sorted(list(b)) if b else ["main"]

    @active_branches.setter
    def active_branches(self, value: List[str]):
        self.metadata["active_branches"] = value

    @property
    def problem_statement(self) -> str:
        return self.task_description

    @problem_statement.setter
    def problem_statement(self, value: str):
        self.task_description = value

    @property
    def ground_truth(self) -> Optional[str]:
        return self.metadata.get("ground_truth")

    @ground_truth.setter
    def ground_truth(self, value: Optional[str]):
        if value is not None:
            self.metadata["ground_truth"] = value
        else:
            self.metadata.pop("ground_truth", None)

    def get_branch_entries(self, branch_id: str = "main") -> List[Any]:
        return sorted(
            [
                c for c in self.contributions
                if (getattr(c, "metadata", {}).get("branch_id") or getattr(c, "branch_id", "main")) == branch_id
            ],
            key=lambda c: getattr(c, "turn_index", getattr(c, "step_number", 0)),
        )

    def get_entry(self, entry_id: str) -> Optional[Any]:
        for c in self.contributions:
            cid = getattr(c, "contribution_id", getattr(c, "entry_id", None))
            if cid == entry_id:
                return c
        return None

    def get_latest_entry(self, branch_id: str = "main") -> Optional[Any]:
        b_entries = self.get_branch_entries(branch_id)
        return b_entries[-1] if b_entries else None


# ============================================================================
# Concurrency & Write-Lock Management (Student 1)
# ============================================================================

class LockAcquireRequest(BaseModel):
    """Request to acquire an exclusive write lock on a blackboard session."""
    model_config = ConfigDict(extra="ignore")

    session_id: str = Field(..., description="Target session ID.")
    agent_id: str = Field(..., description="Agent requesting the lock.")
    timeout_seconds: float = Field(
        default=5.0,
        gt=0.0,
        description="Maximum lock hold duration before automatic TTL expiration."
    )


class LockAcquireResponse(BaseModel):
    """Response returned when an agent attempts to acquire a write lock."""
    model_config = ConfigDict(extra="ignore")

    acquired: bool = Field(..., description="True if the lock was successfully acquired.")
    lock_token: Optional[str] = Field(
        default=None,
        description="Cryptographic / UUID lease token required to execute writes."
    )
    holder_id: Optional[str] = Field(
        default=None,
        description="ID of the agent currently holding the lock if acquisition failed."
    )
    expires_at: Optional[datetime] = Field(
        default=None,
        description="Expiration timestamp for the acquired lock lease."
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error detail if lock acquisition failed or timed out."
    )


class LockReleaseRequest(BaseModel):
    """Request to release a held exclusive write lock."""
    model_config = ConfigDict(extra="ignore")

    session_id: str = Field(..., description="Target session ID.")
    agent_id: str = Field(..., description="Agent releasing the lock.")
    lock_token: str = Field(..., description="Lease token issued during lock acquisition.")


class StateWriteRequest(BaseModel):
    """Transactional write request to commit an agent contribution to the blackboard."""
    model_config = ConfigDict(extra="ignore")

    session_id: str = Field(..., description="Target session ID.")
    agent_id: str = Field(..., description="Contributing agent ID.")
    lock_token: str = Field(..., description="Valid lock token held by the agent.")
    expected_version: int = Field(
        ...,
        ge=0,
        description="Expected current state version for optimistic concurrency control."
    )
    contribution: AgentContribution = Field(
        ...,
        description="The contribution to append to blackboard history."
    )


# ============================================================================
# Deterministic Scheduler & Deadlock Hooks (Student 1)
# ============================================================================

class ScheduledTurn(BaseModel):
    """An agent turn queued by the deterministic scheduler."""
    model_config = ConfigDict(extra="ignore")

    turn_id: str = Field(default_factory=_generate_id, description="Unique turn ID.")
    session_id: str = Field(..., description="Blackboard session ID.")
    agent_id: str = Field(..., description="Agent scheduled to act.")
    priority: int = Field(
        default=1,
        ge=0,
        description="Turn priority (lower integer = higher priority)."
    )
    target_contribution_id: Optional[str] = Field(
        default=None,
        description="Specific contribution the scheduled agent must respond to."
    )
    scheduled_at: datetime = Field(
        default_factory=_utc_now,
        description="Queue insertion timestamp."
    )
    deadline: Optional[datetime] = Field(
        default=None,
        description="Optional turn execution deadline."
    )


class SchedulerQueueState(BaseModel):
    """Current state of the turn arbitration priority queue."""
    model_config = ConfigDict(extra="ignore")

    session_id: str = Field(..., description="Blackboard session ID.")
    current_turn_index: int = Field(..., ge=0, description="Current turn index.")
    active_agent_id: Optional[str] = Field(
        default=None,
        description="Agent currently executing its turn."
    )
    pending_queue: List[ScheduledTurn] = Field(
        default_factory=list,
        description="Ordered list of queued turns."
    )
    is_paused: bool = Field(
        default=False,
        description="True if scheduler is paused (e.g., during deadlock resolution)."
    )


class DeadlockNotification(BaseModel):
    """Payload emitted when the scheduler detects an impasse (consumed by Student 3)."""
    model_config = ConfigDict(extra="ignore")

    session_id: str = Field(..., description="Session where deadlock occurred.")
    deadlock_turn_index: int = Field(..., ge=0, description="Turn index at detection.")
    conflicting_agent_ids: List[str] = Field(
        ...,
        description="Agents involved in the disagreement loop."
    )
    repeated_tags: List[PXPTag] = Field(
        ...,
        description="Sequence of tags triggering the deadlock heuristic (e.g. [REFUTE, REJECT])."
    )
    snapshot: BlackboardSnapshot = Field(
        ...,
        description="Deep-copy snapshot of the blackboard state at impasse."
    )
    detected_at: datetime = Field(
        default_factory=_utc_now,
        description="Timestamp of deadlock detection."
    )
    reason: str = Field(
        ...,
        description="Explanation of the deadlock heuristic trigger."
    )


# ============================================================================
# Retrospective Counterfactual Models (Student 3 Integration)
# ============================================================================

class CreditAssignmentDelta(BaseModel):
    """Blame/credit attribution identifying which past turn precipitated deadlock."""
    model_config = ConfigDict(extra="ignore")

    agent_id: str = Field(..., description="Agent evaluated.")
    turn_index: int = Field(..., ge=0, description="Blackboard turn index evaluated.")
    contribution_id: str = Field(..., description="Target contribution ID.")
    causality_score: float = Field(
        ...,
        description="Quantified causality score for the impasse [-1.0 = highly beneficial, +1.0 = direct cause of deadlock]."
    )
    rationale: str = Field(
        ...,
        description="Counterfactual rationale explaining the credit/blame attribution."
    )


class CounterfactualBranch(BaseModel):
    """An isolated alternative timeline explored in the Retrospective Sandbox."""
    model_config = ConfigDict(extra="ignore")

    branch_id: str = Field(default_factory=_generate_id, description="Unique branch ID.")
    session_id: str = Field(..., description="Parent blackboard session ID.")
    forked_at_turn_index: int = Field(
        ...,
        ge=0,
        description="Turn index in parent history where branch diverged."
    )
    divergence_agent_id: str = Field(
        ...,
        description="Agent whose assertion was altered in this simulation."
    )
    counterfactual_assertion: AgentContribution = Field(
        ...,
        description="The modified hypothetical contribution replacing the original turn."
    )
    simulated_trajectory: List[AgentContribution] = Field(
        default_factory=list,
        description="Subsequent turns simulated within the sandbox."
    )
    outcome_status: BlackboardStatus = Field(
        ...,
        description="Resulting terminal status of this counterfactual branch."
    )
    convergence_score: float = Field(
        ...,
        description="Metric measuring how effectively this branch broke the deadlock."
    )


class ReviseProposal(BaseModel):
    """The winning counterfactual update submitted to the Scheduler to resolve impasse."""
    model_config = ConfigDict(extra="ignore")

    session_id: str = Field(..., description="Blackboard session ID.")
    winning_branch_id: str = Field(
        ...,
        description="Sandbox branch ID that successfully resolved the deadlock."
    )
    target_turn_index: int = Field(
        ...,
        ge=0,
        description="Turn index being revised or amended."
    )
    revised_contribution: AgentContribution = Field(
        ...,
        description="The formatted REVISE contribution to inject into the live blackboard."
    )
    credit_assignment_summary: str = Field(
        ...,
        description="Summary of credit assignment analysis justifying this revision."
    )


# ============================================================================
# Telemetry & Visualization Streaming Models (Student 4 Integration)
# ============================================================================

class TelemetryEvent(BaseModel):
    """Structured telemetry event streamed over WebSockets to the UI frontend."""
    model_config = ConfigDict(extra="ignore")

    event_id: str = Field(default_factory=_generate_id, description="Unique event ID.")
    event_type: TelemetryEventType = Field(..., description="Type of event.")
    session_id: str = Field(..., description="Blackboard session ID.")
    timestamp: datetime = Field(
        default_factory=_utc_now,
        description="Event generation UTC timestamp."
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Event-specific structured data."
    )


# ============================================================================
# Intelligibility Evaluation & Benchmarking Models
# ============================================================================

class IntelligibilityAssessment(BaseModel):
    """Evaluation result assessing Michie-Baskar intelligibility of an interaction."""
    model_config = ConfigDict(extra="ignore")

    session_id: str = Field(..., description="Assessed blackboard session ID.")
    level: IntelligibilityLevel = Field(
        ...,
        description="Assigned intelligibility tier (NONE, ONE_WAY, STRONG, ULTRA_STRONG)."
    )
    is_one_way: bool = Field(
        ...,
        description="True if explanations were comprehensible to the evaluator."
    )
    is_strong: bool = Field(
        ...,
        description="True if every turn adhered to the intelligible PXP state machine."
    )
    is_ultra_strong: bool = Field(
        ...,
        description="True if measurable knowledge transfer / model update occurred."
    )
    tag_counts: Dict[PXPTag, int] = Field(
        default_factory=dict,
        description="Histogram of PXP tags exchanged during the session."
    )
    notes: Optional[str] = Field(
        default=None,
        description="Evaluator observations or qualitative remarks."
    )


class TrialConfig(BaseModel):
    """Configuration for an automated benchmark ablation trial."""
    model_config = ConfigDict(extra="ignore")

    benchmark_name: BenchmarkType = Field(..., description="Benchmark dataset.")
    task_id: str = Field(..., description="Identifier of the task or QA item.")
    counterfactual_density: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of counterfactual-capable agents in the ensemble (0.0, 0.33, 0.66, 1.0)."
    )
    agent_models: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of agent IDs/roles to model names (e.g. Qwen2.5-7B, Llama-3.1-8B)."
    )
    max_turns: int = Field(
        default=15,
        gt=0,
        description="Turn ceiling before terminating trial."
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility."
    )


class TrialResult(BaseModel):
    """Aggregated metrics collected from an execution trial for paper analytics."""
    model_config = ConfigDict(extra="ignore")

    session_id: str = Field(..., description="Session identifier.")
    benchmark_name: BenchmarkType = Field(..., description="Benchmark evaluated.")
    task_id: str = Field(..., description="Task identifier.")
    counterfactual_density: float = Field(..., description="Ablation density tested.")
    final_status: BlackboardStatus = Field(..., description="Terminal state of the session.")
    consensus_reached: bool = Field(..., description="True if consensus was reached.")
    turns_taken: int = Field(..., ge=0, description="Total turns executed.")
    deadlock_count: int = Field(
        default=0,
        ge=0,
        description="Number of deadlocks triggered."
    )
    counterfactual_interventions: int = Field(
        default=0,
        ge=0,
        description="Number of retrospective counterfactual recovery attempts."
    )
    intelligibility: IntelligibilityAssessment = Field(
        ...,
        description="Intelligibility assessment result."
    )
    total_token_cost: int = Field(
        default=0,
        ge=0,
        description="Total tokens consumed across all agent calls."
    )
    duration_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Total wall-clock duration in seconds."
    )


class BlackboardState(BlackboardSnapshot):
    """
    Unified blackboard state model bridging BlackboardSnapshot with
    legacy constructor parameters and methods for S2, S3, S4 modules.
    """
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            d = dict(data)
            if "task_description" not in d:
                d["task_description"] = (
                    d.get("problem_statement")
                    or d.get("task_prompt")
                    or f"Task {d.get('task_id', 'unknown')}"
                )
            status = d.get("status")
            if isinstance(status, str):
                status_map = {
                    "IN_PROGRESS": BlackboardStatus.ACTIVE,
                    "ACTIVE": BlackboardStatus.ACTIVE,
                    "CONSENSUS": BlackboardStatus.CONSENSUS,
                    "DEADLOCK": BlackboardStatus.DEADLOCK,
                    "RESOLVED": BlackboardStatus.RESOLVED,
                    "INITIALIZING": BlackboardStatus.INITIALIZING,
                    "TERMINATED": BlackboardStatus.TERMINATED,
                }
                d["status"] = status_map.get(status.upper(), BlackboardStatus.ACTIVE)
            meta = dict(d.get("metadata") or {})
            if "ground_truth" in d and "ground_truth" not in meta:
                meta["ground_truth"] = d["ground_truth"]
            if "active_branches" in d and "active_branches" not in meta:
                meta["active_branches"] = d["active_branches"]
            d["metadata"] = meta
            if "contributions" not in d and "entries" in d:
                session_id = d.get("session_id", "main")
                contribs = []
                for e in d["entries"]:
                    if isinstance(e, AgentContribution):
                        contribs.append(e)
                    elif hasattr(e, "to_agent_contribution"):
                        contribs.append(e.to_agent_contribution(session_id=session_id))
                    elif isinstance(e, dict):
                        contribs.append(
                            BlackboardEntry.model_validate(e).to_agent_contribution(session_id=session_id)
                        )
                d["contributions"] = contribs
            return d
        return data

    def __init__(
        self,
        session_id: str,
        task_id: str,
        task_description: Optional[str] = None,
        problem_statement: Optional[str] = None,
        ground_truth: Optional[str] = None,
        status: Union[BlackboardStatus, str] = BlackboardStatus.ACTIVE,
        entries: Optional[List[Any]] = None,
        contributions: Optional[List[AgentContribution]] = None,
        active_branches: Optional[List[str]] = None,
        intelligibility_level: Optional[Any] = None,
        **kwargs: Any,
    ):
        desc = task_description or problem_statement or f"Task {task_id}"
        if isinstance(status, str):
            status_map = {
                "IN_PROGRESS": BlackboardStatus.ACTIVE,
                "ACTIVE": BlackboardStatus.ACTIVE,
                "CONSENSUS": BlackboardStatus.CONSENSUS,
                "DEADLOCK": BlackboardStatus.DEADLOCK,
                "RESOLVED": BlackboardStatus.RESOLVED,
                "INITIALIZING": BlackboardStatus.INITIALIZING,
                "TERMINATED": BlackboardStatus.TERMINATED,
            }
            mapped_status = status_map.get(status.upper(), BlackboardStatus.ACTIVE)
        else:
            mapped_status = status

        meta = kwargs.get("metadata", {})
        if ground_truth:
            meta["ground_truth"] = ground_truth
        if active_branches:
            meta["active_branches"] = active_branches

        contrib_list: List[AgentContribution] = []
        if contributions:
            contrib_list = contributions
        elif entries:
            for e in entries:
                if isinstance(e, AgentContribution):
                    contrib_list.append(e)
                elif hasattr(e, "to_agent_contribution"):
                    contrib_list.append(e.to_agent_contribution(session_id=session_id))
                elif isinstance(e, dict):
                    contrib_list.append(BlackboardEntry.model_validate(e).to_agent_contribution(session_id=session_id))

        super().__init__(
            session_id=session_id,
            task_id=task_id,
            task_description=desc,
            status=mapped_status,
            contributions=contrib_list,
            metadata=meta,
            **kwargs,
        )


StreamEvent = TelemetryEvent


class BlackboardEntry(BaseModel):
    """
    Unified entry model bridging flat Blackboard entry representation
    with Student 1's nested PEXPayload and AgentContribution specification.
    """
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    entry_id: str = Field(default_factory=_generate_id, description="Unique contribution identifier")
    parent_id: Optional[str] = Field(default=None, description="Target contribution ID")
    branch_id: str = Field(default="main", description="Branch identifier")
    agent_id: str = Field(..., description="Contributing agent ID")
    agent_role: str = Field(default="generalist", description="Agent role")
    tag: PXPTag = Field(..., description="PXP operational tag")
    prediction: str = Field(..., description="Hypothesis claim")
    explanation: str = Field(..., description="Reasoning rationale")
    evidence_refs: List[str] = Field(default_factory=list, description="Evidence citations")
    confidence: float = Field(default=0.9, ge=0.0, le=1.0, description="Calibrated confidence")
    step_number: int = Field(default=1, ge=0, description="Turn index")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary")

    @field_validator("prediction", "explanation")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field must not be empty or whitespace.")
        return v.strip()

    def to_agent_contribution(self, session_id: str = "main") -> AgentContribution:
        """Convert to Student 1's nested AgentContribution schema."""
        role = None
        if self.agent_role:
            try:
                role = AgentRole(self.agent_role.upper())
            except ValueError:
                role = None
        meta = dict(self.metadata)
        meta["branch_id"] = self.branch_id
        meta["agent_role_detail"] = self.agent_role
        return AgentContribution(
            contribution_id=self.entry_id,
            session_id=session_id,
            turn_index=self.step_number,
            agent_id=self.agent_id,
            agent_role=role,
            tag=self.tag,
            target_contribution_id=self.parent_id,
            payload=PEXPayload(
                prediction=Prediction(
                    claim=self.prediction,
                    confidence=self.confidence,
                ),
                explanation=Explanation(
                    rationale=self.explanation,
                    evidence=self.evidence_refs,
                ),
            ),
            is_counterfactual=self.metadata.get("is_simulated", False),
            metadata=meta,
        )

    @classmethod
    def from_agent_contribution(cls, contrib: AgentContribution) -> "BlackboardEntry":
        """Construct from Student 1's AgentContribution schema."""
        role_str = (
            contrib.metadata.get("agent_role_detail")
            or (contrib.agent_role.value if contrib.agent_role else "generalist")
        )
        pred_claim = contrib.payload.prediction.claim if (contrib.payload and contrib.payload.prediction) else ""
        expl_rationale = contrib.payload.explanation.rationale if (contrib.payload and contrib.payload.explanation) else ""
        ev_refs = (contrib.payload.explanation.evidence or []) if (contrib.payload and contrib.payload.explanation) else []
        conf = contrib.payload.prediction.confidence if (contrib.payload and contrib.payload.prediction) else 0.0

        return cls(
            entry_id=contrib.contribution_id,
            parent_id=contrib.target_contribution_id,
            branch_id=contrib.metadata.get("branch_id", "main"),
            agent_id=contrib.agent_id,
            agent_role=role_str,
            tag=contrib.tag,
            prediction=pred_claim,
            explanation=expl_rationale,
            evidence_refs=ev_refs,
            confidence=conf,
            step_number=contrib.turn_index,
            metadata=contrib.metadata,
        )

