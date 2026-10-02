"""
contracts package exports for the Intelligible Blackboard Architecture.
Grounded in Baskar et al. (2025) Two-Way Intelligibility Protocol (PXP).
"""

from .schemas import (
    # Core Protocol Enums
    PXPTag,
    BlackboardStatus,
    IntelligibilityLevel,
    AgentRole,
    BenchmarkType,
    TelemetryEventType,
    # PEX Prediction & Explanation
    Prediction,
    Explanation,
    PEXPayload,
    # Contributions & Blackboard State
    AgentContribution,
    BlackboardSnapshot,
    # Concurrency & Write Lock
    LockAcquireRequest,
    LockAcquireResponse,
    LockReleaseRequest,
    StateWriteRequest,
    # Scheduler & Deadlocks
    ScheduledTurn,
    SchedulerQueueState,
    DeadlockNotification,
    # Student 3 Counterfactual Integration
    CreditAssignmentDelta,
    CounterfactualBranch,
    ReviseProposal,
    # Student 4 Telemetry & Benchmarking
    TelemetryEvent,
    IntelligibilityAssessment,
    TrialConfig,
    TrialResult,
    # Unified Compatibility Models for S2, S3, S4
    BlackboardEntry,
    BlackboardState,
    StreamEvent,
    BenchmarkTask,
)

__all__ = [
    "PXPTag",
    "BlackboardStatus",
    "IntelligibilityLevel",
    "AgentRole",
    "BenchmarkType",
    "TelemetryEventType",
    "Prediction",
    "Explanation",
    "PEXPayload",
    "AgentContribution",
    "BlackboardSnapshot",
    "LockAcquireRequest",
    "LockAcquireResponse",
    "LockReleaseRequest",
    "StateWriteRequest",
    "ScheduledTurn",
    "SchedulerQueueState",
    "DeadlockNotification",
    "CreditAssignmentDelta",
    "CounterfactualBranch",
    "ReviseProposal",
    "TelemetryEvent",
    "IntelligibilityAssessment",
    "TrialConfig",
    "TrialResult",
    "BlackboardEntry",
    "BlackboardState",
    "StreamEvent",
    "BenchmarkTask",
]
