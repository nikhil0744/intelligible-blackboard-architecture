"""Tests for Pydantic v2 schemas and PXP protocol tags."""

from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from contracts.schemas import (
    AgentContribution,
    AgentRole,
    BenchmarkType,
    BlackboardSnapshot,
    BlackboardStatus,
    CreditAssignmentDelta,
    CounterfactualBranch,
    DeadlockNotification,
    Explanation,
    IntelligibilityAssessment,
    IntelligibilityLevel,
    LockAcquireRequest,
    LockAcquireResponse,
    PEXPayload,
    Prediction,
    PXPTag,
    ReviseProposal,
    ScheduledTurn,
    SchedulerQueueState,
    StateWriteRequest,
    TelemetryEvent,
    TelemetryEventType,
    TrialConfig,
    TrialResult,
    BlackboardState,
    BlackboardEntry,
)


def test_pxp_protocol_tags():
    """Verify the four operational PXP tags defined in Baskar et al."""
    expected_tags = {"RATIFY", "REVISE", "REFUTE", "REJECT"}
    actual_tags = {tag.value for tag in PXPTag}
    assert actual_tags == expected_tags


def test_intelligibility_levels():
    """Verify Michie-Baskar intelligibility tiers."""
    expected_levels = {"NONE", "ONE_WAY", "STRONG", "ULTRA_STRONG"}
    actual_levels = {lvl.value for lvl in IntelligibilityLevel}
    assert actual_levels == expected_levels


def test_pex_payload_creation_and_serialization():
    """Verify Prediction and Explanation models create a valid PEXPayload."""
    pred = Prediction(
        claim="Diagnosis: Community-Acquired Pneumonia",
        confidence=0.88,
        probabilities={"pneumonia": 0.88, "bronchitis": 0.12},
        summary="Patient exhibits symptoms typical of acute pneumonia.",
        metadata={"icd10": "J18.9"}
    )
    expl = Explanation(
        rationale="Chest X-ray shows right lower lobe consolidation with fever and elevated WBC.",
        evidence=["Right lower lobe infiltrate on CXR", "WBC count 14,200/uL", "Temperature 38.6 C"],
        assumptions=["Standard community exposure, non-immunocompromised host"],
        limitations="Cannot completely rule out early aspiration pneumonia"
    )
    pex = PEXPayload(prediction=pred, explanation=expl)

    # Test serialization & round-trip
    json_data = pex.model_dump_json()
    reconstructed = PEXPayload.model_validate_json(json_data)
    assert reconstructed.prediction.claim == pred.claim
    assert reconstructed.explanation.evidence == expl.evidence


def test_prediction_confidence_bounds():
    """Verify confidence must be bounded between 0.0 and 1.0."""
    with pytest.raises(ValidationError):
        Prediction(claim="Test", confidence=1.5)

    with pytest.raises(ValidationError):
        Prediction(claim="Test", confidence=-0.1)


def test_agent_contribution_validation():
    """Verify AgentContribution creation with PXP tag and target reference."""
    payload = PEXPayload(
        prediction=Prediction(claim="Proposal A", confidence=0.75),
        explanation=Explanation(rationale="Initial hypothesis based on preliminary lab results.")
    )
    contrib = AgentContribution(
        session_id="session-101",
        turn_index=0,
        agent_id="agent-diagnostician",
        agent_role=AgentRole.PRIMARY,
        tag=PXPTag.REVISE,
        payload=payload
    )
    assert contrib.tag == PXPTag.REVISE
    assert contrib.turn_index == 0
    assert contrib.session_id == "session-101"
    assert contrib.contribution_id is not None
    assert contrib.timestamp.tzinfo is not None


def test_blackboard_snapshot():
    """Verify BlackboardSnapshot state container."""
    snapshot = BlackboardSnapshot(
        session_id="sess-001",
        task_id="medagent-case-42",
        task_description="Evaluate fever and chest discomfort in 54yo male.",
        status=BlackboardStatus.ACTIVE,
        version=1,
        active_agents=["agent-1", "agent-2"]
    )
    assert snapshot.status == BlackboardStatus.ACTIVE
    assert snapshot.version == 1
    assert len(snapshot.contributions) == 0


def test_write_lock_concurrency_contracts():
    """Verify lock acquire and release contracts for Student 1."""
    req = LockAcquireRequest(session_id="sess-001", agent_id="agent-1", timeout_seconds=3.0)
    assert req.timeout_seconds == 3.0

    resp = LockAcquireResponse(
        acquired=True,
        lock_token="lease-token-abc-123",
        holder_id="agent-1"
    )
    assert resp.acquired is True
    assert resp.lock_token == "lease-token-abc-123"


def test_deadlock_notification_and_counterfactual():
    """Verify DeadlockNotification contract used between S1 and S3."""
    snapshot = BlackboardSnapshot(
        session_id="sess-deadlock",
        task_id="krama-12",
        task_description="Engineering constraint resolution.",
        status=BlackboardStatus.DEADLOCK,
        version=5
    )
    deadlock = DeadlockNotification(
        session_id="sess-deadlock",
        deadlock_turn_index=4,
        conflicting_agent_ids=["agent-A", "agent-B"],
        repeated_tags=[PXPTag.REFUTE, PXPTag.REJECT, PXPTag.REFUTE],
        snapshot=snapshot,
        reason="Circular REFUTE/REJECT loop detected between agent-A and agent-B"
    )
    assert deadlock.session_id == "sess-deadlock"
    assert deadlock.repeated_tags == [PXPTag.REFUTE, PXPTag.REJECT, PXPTag.REFUTE]


def test_telemetry_event_streaming():
    """Verify TelemetryEvent format for Student 4 WebSocket broadcasting."""
    event = TelemetryEvent(
        event_type=TelemetryEventType.DEADLOCK_DETECTED,
        session_id="sess-001",
        payload={"turn": 4, "reason": "REFUTE loop"}
    )
    assert event.event_type == TelemetryEventType.DEADLOCK_DETECTED
    assert event.payload["turn"] == 4


def test_blackboard_state_model_validate_from_dict():
    """Verify BlackboardState.model_validate normalizes legacy fields from dictionary fixtures."""
    raw_fixture = {
        "session_id": "sess-raw-fixture",
        "task_id": "med-001",
        "problem_statement": "Patient presents with persistent cough and dyspnea.",
        "ground_truth": "IPF",
        "status": "IN_PROGRESS",
        "active_branches": ["main", "sandbox_1"],
        "entries": [
            {
                "entry_id": "e1",
                "agent_id": "agent_pulmo",
                "agent_role": "Pulmonologist",
                "tag": "RATIFY",
                "prediction": "IPF",
                "explanation": "Honeycombing on HRCT",
                "evidence_refs": ["HRCT-01"],
                "step_number": 1,
            }
        ],
    }
    state = BlackboardState.model_validate(raw_fixture)
    assert state.session_id == "sess-raw-fixture"
    assert state.problem_statement == "Patient presents with persistent cough and dyspnea."
    assert state.ground_truth == "IPF"
    assert state.status == BlackboardStatus.ACTIVE
    assert "sandbox_1" in state.active_branches
    assert len(state.contributions) == 1
    assert state.contributions[0].contribution_id == "e1"


def test_agent_contribution_empty_evidence_and_setters():
    """Verify AgentContribution handles empty evidence without AttributeError and supports setters."""
    contrib = AgentContribution(
        contribution_id="c1",
        session_id="s1",
        turn_index=1,
        agent_id="agent_1",
        tag=PXPTag.RATIFY,
        payload=PEXPayload(
            prediction=Prediction(claim="Test Claim", confidence=0.8),
            explanation=Explanation(rationale="Test Rationale", evidence=[]),
        ),
    )
    # Empty evidence must return empty list without crashing on non-existent prediction.grounding_references
    assert contrib.evidence_refs == []

    # Test property setters
    contrib.step_number = 5
    assert contrib.turn_index == 5
    assert contrib.step_number == 5

    contrib.branch_id = "sandbox_branch_42"
    assert contrib.branch_id == "sandbox_branch_42"
    assert contrib.metadata["branch_id"] == "sandbox_branch_42"

    contrib.parent_id = "c0"
    assert contrib.parent_id == "c0"
    assert contrib.target_contribution_id == "c0"


def test_persona_role_preservation():
    """Verify specialized persona roles (e.g. Cardiologist) survive conversion roundtrips."""
    entry = BlackboardEntry(
        entry_id="e2",
        agent_id="agent_cardio",
        agent_role="Cardiologist",
        tag=PXPTag.REVISE,
        prediction="Congestive Heart Failure",
        explanation="Elevated BNP and bilateral edema",
        evidence_refs=["BNP-900"],
    )
    contrib = entry.to_agent_contribution(session_id="s_med")
    assert contrib.metadata.get("agent_role_detail") == "Cardiologist"

    reconstructed = BlackboardEntry.from_agent_contribution(contrib)
    assert reconstructed.agent_role == "Cardiologist"
    assert reconstructed.prediction == "Congestive Heart Failure"

