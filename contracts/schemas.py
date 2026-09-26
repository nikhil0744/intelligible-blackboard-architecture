"""Data contracts and schemas for the Intelligible Blackboard Architecture.

This module defines Pydantic v2 schemas for the multi-agent blackboard system,
grounded in the PXP (Prediction-eXplanation Protocol) by Baskar et al. (2025)
and Donald Michie's framework for machine intelligibility.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
