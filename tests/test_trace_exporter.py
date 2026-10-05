"""
tests/test_trace_exporter.py
Comprehensive unit tests for the telemetry trace exporter.
"""

import json
from pathlib import Path
import pytest

from agents.board_client import InMemoryBoard
from contracts.schemas import (
    AgentContribution,
    BlackboardSnapshot,
    BlackboardStatus,
    Explanation,
    PEXPayload,
    PXPTag,
    Prediction,
)
from streaming.trace_exporter import (
    export_from_board,
    export_from_spike_blackboard,
    export_snapshot,
)


def test_export_snapshot_empty_contributions(tmp_path: Path):
    """Snapshot with 0 turns should output valid Step 0 and not crash."""
    snap = BlackboardSnapshot(
        session_id="sess_empty",
        task_id="task_00",
        task_description="Solve x + y = 10",
        status=BlackboardStatus.ACTIVE,
    )
    out_file = tmp_path / "trace_empty.json"
    data = export_snapshot(snap, output_path=str(out_file))

    assert out_file.exists()
    assert data["sessionId"] == "sess_empty"
    assert len(data["steps"]) == 1
    assert data["steps"][0]["step"] == 0
    assert data["steps"][0]["turn"] == "T0 • INIT"
    assert len(data["agents"]) == 3  # Padded to 3


def test_export_snapshot_single_turn_and_none_fields(tmp_path: Path):
    """Snapshot with null token usage and null latency should format cleanly."""
    snap = BlackboardSnapshot(
        session_id="sess_single",
        task_id="task_single",
        task_description="Diagnose patient with cough",
        status=BlackboardStatus.ACTIVE,
    )
    c = AgentContribution(
        session_id=snap.session_id,
        turn_index=0,
        agent_id="clinician_0",
        tag=PXPTag.REVISE,
        payload=PEXPayload(
            prediction=Prediction(claim="Viral bronchitis", confidence=0.85),
            explanation=Explanation(rationale="Dry cough and normal vitals."),
        ),
        token_usage=None,
        latency_ms=None,
    )
    snap.contributions.append(c)

    out_file = tmp_path / "trace_single.json"
    data = export_snapshot(snap, output_path=str(out_file))

    assert len(data["steps"]) == 2  # Step 0 + Step 1
    step1 = data["steps"][1]
    assert step1["tokens"] == "0 tok"
    assert step1["claim"] == "Viral bronchitis"
    assert step1["deadlock"] is False


def test_export_snapshot_latency_unit_scaling(tmp_path: Path):
    """Latency in milliseconds (e.g. 2500ms) should format as seconds (2.5s)."""
    snap = BlackboardSnapshot(
        session_id="sess_latency",
        task_id="task_lat",
        task_description="Test latency scaling",
        status=BlackboardStatus.ACTIVE,
    )
    c = AgentContribution(
        session_id=snap.session_id,
        turn_index=0,
        agent_id="clinician_0",
        tag=PXPTag.REVISE,
        payload=PEXPayload(
            prediction=Prediction(claim="Claim", confidence=0.9),
            explanation=Explanation(rationale="Rationale"),
        ),
        token_usage=1200,
        latency_ms=2500.0,
    )
    snap.contributions.append(c)

    out_file = tmp_path / "trace_lat.json"
    data = export_snapshot(snap, output_path=str(out_file))

    step1 = data["steps"][1]
    assert "2.5s" in step1["tokens"]
    assert "2500" not in step1["tokens"]


def test_export_snapshot_deadlock_detection(tmp_path: Path):
    """Two consecutive REFUTE/REJECT contributions must trigger deadlock."""
    snap = BlackboardSnapshot(
        session_id="sess_deadlock",
        task_id="task_dl",
        task_description="Contested diagnosis",
        status=BlackboardStatus.DEADLOCK,
    )
    c1 = AgentContribution(
        session_id=snap.session_id,
        turn_index=0,
        agent_id="agent_a",
        tag=PXPTag.REVISE,
        payload=PEXPayload(
            prediction=Prediction(claim="Option A", confidence=0.8),
            explanation=Explanation(rationale="Initial hypothesis"),
        ),
    )
    c2 = AgentContribution(
        session_id=snap.session_id,
        turn_index=1,
        agent_id="agent_b",
        tag=PXPTag.REFUTE,
        target_contribution_id=c1.contribution_id,
        payload=PEXPayload(
            prediction=Prediction(claim="Option B", confidence=0.85),
            explanation=Explanation(rationale="Counter hypothesis"),
        ),
    )
    c3 = AgentContribution(
        session_id=snap.session_id,
        turn_index=2,
        agent_id="agent_a",
        tag=PXPTag.REJECT,
        target_contribution_id=c2.contribution_id,
        payload=PEXPayload(
            prediction=Prediction(claim="Reject B, Option A is correct", confidence=0.9),
            explanation=Explanation(rationale="Rejecting counter argument"),
        ),
    )
    snap.contributions.extend([c1, c2, c3])

    out_file = tmp_path / "trace_dl.json"
    data = export_snapshot(snap, output_path=str(out_file))

    # Turn 3 should be flagged as deadlock
    step3 = data["steps"][3]
    assert step3["deadlock"] is True
    assert "Circular" in step3["deadlockMsg"]
    assert data["sandbox"]["branch"].startswith("ISOLATED COUNTERFACTUAL")


def test_export_snapshot_more_than_three_agents(tmp_path: Path):
    """Panels with 4+ agents should not crash and map active slots safely."""
    snap = BlackboardSnapshot(
        session_id="sess_many",
        task_id="task_many",
        task_description="Multi-specialist panel",
        status=BlackboardStatus.ACTIVE,
    )
    for i in range(5):
        snap.contributions.append(
            AgentContribution(
                session_id=snap.session_id,
                turn_index=i,
                agent_id=f"specialist_{i}",
                tag=PXPTag.REVISE,
                payload=PEXPayload(
                    prediction=Prediction(claim=f"Claim {i}", confidence=0.7),
                    explanation=Explanation(rationale=f"Rationale {i}"),
                ),
            )
        )

    out_file = tmp_path / "trace_many.json"
    data = export_snapshot(snap, output_path=str(out_file))

    assert len(data["agents"]) == 3
    # Contributions by agent 0, 1, 2 should map to 0, 1, 2; agents 3 & 4 should be None
    assert data["steps"][1]["agentActive"] == 0
    assert data["steps"][2]["agentActive"] == 1
    assert data["steps"][3]["agentActive"] == 2
    assert data["steps"][4]["agentActive"] is None
    assert data["steps"][5]["agentActive"] is None


def test_export_from_board_with_status_override(tmp_path: Path):
    """export_from_board should respect status_override."""
    board = InMemoryBoard()
    snap = board.create_session("task_board", "Evaluate renal dosing")
    c = AgentContribution(
        session_id=snap.session_id,
        turn_index=0,
        agent_id="clinician_0",
        tag=PXPTag.REVISE,
        payload=PEXPayload(
            prediction=Prediction(claim="Stop drug", confidence=0.9),
            explanation=Explanation(rationale="eGFR < 30"),
        ),
    )
    board.submit(c)

    out_file = tmp_path / "trace_board.json"
    data = export_from_board(
        board,
        snap.session_id,
        output_path=str(out_file),
        status_override=BlackboardStatus.CONSENSUS,
    )

    final_step = data["steps"][-1]
    assert final_step["status"] == "CONSENSUS REACHED"
    assert final_step["tag"] == "RATIFY"


def test_export_from_spike_blackboard(tmp_path: Path):
    """Test exporting from spike fixture blackboard."""
    from blackboard.store import InMemoryBlackboard
    from fixtures.spike_starter import files

    fixture_file = files("fixtures").joinpath("sample_deadlocks.json")
    with fixture_file.open("r") as f:
        raw_data = json.load(f)

    board = InMemoryBlackboard(
        session_id=raw_data["session_id"],
        task_id=raw_data["task_id"],
        problem_statement=raw_data["problem_statement"],
        ground_truth=raw_data["ground_truth"],
    )
    from contracts.schemas import BlackboardEntry
    for entry_dict in raw_data["entries"]:
        board.add_entry(BlackboardEntry(**entry_dict))

    out_file = tmp_path / "trace_spike.json"
    data = export_from_spike_blackboard(board, output_path=str(out_file))

    assert out_file.exists()
    assert data["title"] == "Live Run: Spike Deadlock Recovery"
    assert len(data["steps"]) >= 5
