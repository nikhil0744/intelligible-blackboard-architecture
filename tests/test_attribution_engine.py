"""
tests/test_attribution_engine.py
Unit tests for the Deadlock Attribution & Divergence Engine.
Verifies:
1. Deadlock detection on the clinical fixture (fixtures/sample_deadlocks.json).
2. Backtracking to divergence / trigger nodes via parent_id links.
3. Consensus checkpoint identification (latest vs fallback checkpoint steps).
4. False-positive resistance on constructive debate.
5. Multi-agent (3-agent) circular dispute handling.
"""

import json
from pathlib import Path
from contracts.schemas import BlackboardEntry, BlackboardState, PXPTag
from blackboard.store import InMemoryBlackboard
from counterfactual.attribution import (
    AttributionResult,
    DeadlockAttributionEngine,
)


def load_clinical_fixture() -> BlackboardState:
    fixture_path = (
        Path(__file__).resolve().parent.parent
        / "fixtures"
        / "sample_deadlocks.json"
    )
    with open(fixture_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return BlackboardState.model_validate(data)


def test_detects_deadlock_on_clinical_fixture():
    """Verifies deadlock detection and diagnostic extraction on sample_deadlocks.json."""
    state = load_clinical_fixture()
    engine = DeadlockAttributionEngine()

    # Test is_deadlocked directly
    assert engine.is_deadlocked(state.entries, window=2) is True

    # Test analyze() on BlackboardState
    result: AttributionResult = engine.analyze(state)

    assert result.deadlock_detected is True
    assert set(result.conflicting_agents) == {"agent_cardio", "agent_pulmo"}
    assert "e3" in result.conflict_loop_entries
    assert "e4" in result.conflict_loop_entries
    assert result.divergence_entry_id == "e1"
    assert result.rollback_step == 1
    assert (
        result.contested_predictions.get("agent_cardio")
        == "Congestive Heart Failure (CHF)"
    )
    assert (
        result.contested_predictions.get("agent_pulmo")
        == "Idiopathic Pulmonary Fibrosis (IPF)"
    )
    assert len(result.clashing_evidence) > 0
    assert "honeycombing" in result.clashing_evidence
    assert result.attribution_confidence > 0.0
    assert len(result.reason) > 0

    # Test with InMemoryBlackboard
    board = InMemoryBlackboard(
        session_id=state.session_id,
        task_id=state.task_id,
        problem_statement=state.problem_statement,
        ground_truth=state.ground_truth,
    )
    for entry in state.entries:
        board.add_entry(entry)

    result_board = engine.analyze(board)
    assert result_board.deadlock_detected is True
    assert result_board.divergence_entry_id == "e1"
    assert result_board.rollback_step == 1


def test_backtracks_to_divergence_node():
    """Verifies climbing parent_id links back to the proposal that triggered divergence."""
    engine = DeadlockAttributionEngine()

    entries = [
        BlackboardEntry(
            entry_id="e1",
            parent_id=None,
            branch_id="main",
            agent_id="agent_cardio",
            agent_role="Cardiologist",
            tag=PXPTag.PROPOSE,
            prediction="Initial condition: Stable",
            explanation="Baseline hemodynamic indicators within limits.",
            step_number=1,
        ),
        BlackboardEntry(
            entry_id="e2",
            parent_id="e1",
            branch_id="main",
            agent_id="agent_pulmo",
            agent_role="Pulmonologist",
            tag=PXPTag.RATIFY,
            prediction="Initial condition: Stable",
            explanation="Agree with baseline stability.",
            step_number=2,
        ),
        BlackboardEntry(
            entry_id="e3",
            parent_id="e2",
            branch_id="main",
            agent_id="agent_cardio",
            agent_role="Cardiologist",
            tag=PXPTag.PROPOSE,
            prediction="Secondary Diagnosis: Hypertension",
            explanation="Elevated systolic readings indicate essential hypertension.",
            step_number=3,
        ),
        BlackboardEntry(
            entry_id="e4",
            parent_id="e3",
            branch_id="main",
            agent_id="agent_neuro",
            agent_role="Neurologist",
            tag=PXPTag.REFUTE,
            prediction="Secondary Diagnosis: Pheochromocytoma",
            explanation="Paroxysmal hypertension and headache suggest pheochromocytoma.",
            evidence_refs=["headache", "paroxysm"],
            step_number=4,
        ),
        BlackboardEntry(
            entry_id="e5",
            parent_id="e4",
            branch_id="main",
            agent_id="agent_cardio",
            agent_role="Cardiologist",
            tag=PXPTag.REJECT,
            prediction="Secondary Diagnosis: Hypertension",
            explanation="Pheochromocytoma is exceptionally rare; essential hypertension is far more likely.",
            evidence_refs=["prevalence_stats"],
            step_number=5,
        ),
        BlackboardEntry(
            entry_id="e6",
            parent_id="e5",
            branch_id="main",
            agent_id="agent_neuro",
            agent_role="Neurologist",
            tag=PXPTag.REJECT,
            prediction="Secondary Diagnosis: Pheochromocytoma",
            explanation="Plasma metanephrines are 3x normal, ruling out essential hypertension.",
            evidence_refs=["metanephrines"],
            step_number=6,
        ),
    ]

    div_id, rollback_step, climb_path = engine.locate_divergence_node(entries)
    assert div_id == "e3"
    assert rollback_step == 3
    assert "e4" in climb_path
    assert "e3" in climb_path

    result = engine.analyze(entries)
    assert result.deadlock_detected is True
    assert result.divergence_entry_id == "e3"
    assert result.rollback_step == 3
    assert set(result.conflicting_agents) == {"agent_cardio", "agent_neuro"}
    assert "e4" in result.conflict_loop_entries
    assert "e5" in result.conflict_loop_entries
    assert "e6" in result.conflict_loop_entries


def test_identifies_consensus_checkpoints():
    """Verifies finding ratified consensus checkpoints and extracting latest vs fallback checkpoints."""
    engine = DeadlockAttributionEngine()

    entries = [
        BlackboardEntry(
            entry_id="e1",
            parent_id=None,
            branch_id="main",
            agent_id="agent_A",
            tag=PXPTag.PROPOSE,
            prediction="Diagnosis Alpha",
            explanation="Initial proposal for syndrome A.",
            step_number=1,
        ),
        BlackboardEntry(
            entry_id="e2",
            parent_id="e1",
            branch_id="main",
            agent_id="agent_B",
            tag=PXPTag.RATIFY,
            prediction="Diagnosis Alpha",
            explanation="Align with Diagnosis Alpha rationale.",
            step_number=2,
        ),
        BlackboardEntry(
            entry_id="e3",
            parent_id="e2",
            branch_id="main",
            agent_id="agent_A",
            tag=PXPTag.PROPOSE,
            prediction="Treatment Beta",
            explanation="Recommended primary therapy.",
            step_number=3,
        ),
        BlackboardEntry(
            entry_id="e4",
            parent_id="e3",
            branch_id="main",
            agent_id="agent_B",
            tag=PXPTag.RATIFY,
            prediction="Treatment Beta",
            explanation="Align with Treatment Beta recommendations.",
            step_number=4,
        ),
        BlackboardEntry(
            entry_id="e5",
            parent_id="e4",
            branch_id="main",
            agent_id="agent_A",
            tag=PXPTag.PROPOSE,
            prediction="Discharge Protocol Gamma",
            explanation="Propose 24h discharge protocol.",
            step_number=5,
        ),
        BlackboardEntry(
            entry_id="e6",
            parent_id="e5",
            branch_id="main",
            agent_id="agent_B",
            tag=PXPTag.REFUTE,
            prediction="Discharge Protocol Delta",
            explanation="Patient requires 72h observation.",
            step_number=6,
        ),
        BlackboardEntry(
            entry_id="e7",
            parent_id="e6",
            branch_id="main",
            agent_id="agent_A",
            tag=PXPTag.REJECT,
            prediction="Discharge Protocol Gamma",
            explanation="Observation unnecessary given biomarker stability.",
            step_number=7,
        ),
        BlackboardEntry(
            entry_id="e8",
            parent_id="e7",
            branch_id="main",
            agent_id="agent_B",
            tag=PXPTag.REJECT,
            prediction="Discharge Protocol Delta",
            explanation="Risk of late rebound too high.",
            step_number=8,
        ),
    ]

    checkpoints = engine.find_consensus_checkpoints(entries)
    assert len(checkpoints) == 2
    assert checkpoints[0].entry_id == "e2"
    assert checkpoints[0].step_number == 2
    assert checkpoints[1].entry_id == "e4"
    assert checkpoints[1].step_number == 4

    result = engine.analyze(entries)
    assert result.deadlock_detected is True
    assert result.latest_checkpoint_id == "e4"
    assert result.latest_checkpoint_step == 4
    assert result.fallback_checkpoint_id == "e2"
    assert result.fallback_checkpoint_step == 2
    assert result.divergence_entry_id == "e5"
    assert result.rollback_step == 5


def test_returns_no_deadlock_on_constructive_debate():
    """Verifies that constructive debate with revisions and ratification does NOT trigger deadlock."""
    engine = DeadlockAttributionEngine()

    entries = [
        BlackboardEntry(
            entry_id="e1",
            parent_id=None,
            branch_id="main",
            agent_id="agent_cardio",
            agent_role="Cardiologist",
            tag=PXPTag.PROPOSE,
            prediction="Congestive Heart Failure (CHF)",
            explanation="Dyspnea and crackles point to fluid overload.",
            step_number=1,
        ),
        BlackboardEntry(
            entry_id="e2",
            parent_id="e1",
            branch_id="main",
            agent_id="agent_pulmo",
            agent_role="Pulmonologist",
            tag=PXPTag.REFUTE,
            prediction="Idiopathic Pulmonary Fibrosis (IPF)",
            explanation="HRCT shows honeycombing; normal BNP rules out CHF.",
            step_number=2,
        ),
        BlackboardEntry(
            entry_id="e3",
            parent_id="e2",
            branch_id="main",
            agent_id="agent_cardio",
            agent_role="Cardiologist",
            tag=PXPTag.REVISE,
            prediction="Idiopathic Pulmonary Fibrosis with Secondary Pulmonary Hypertension",
            explanation="Concur with HRCT findings. Revising to IPF with secondary cardiopulmonary involvement.",
            step_number=3,
        ),
        BlackboardEntry(
            entry_id="e4",
            parent_id="e3",
            branch_id="main",
            agent_id="agent_pulmo",
            agent_role="Pulmonologist",
            tag=PXPTag.RATIFY,
            prediction="Idiopathic Pulmonary Fibrosis with Secondary Pulmonary Hypertension",
            explanation="Fully ratify revised composite diagnosis.",
            step_number=4,
        ),
    ]

    assert engine.is_deadlocked(entries, window=2) is False

    result = engine.analyze(entries)
    assert result.deadlock_detected is False
    assert result.conflicting_agents == []
    assert result.conflict_loop_entries == []
    assert result.divergence_entry_id is None
    assert result.rollback_step == 0
    assert result.attribution_confidence == 0.0
    assert result.latest_checkpoint_id == "e4"
    assert result.latest_checkpoint_step == 4


def test_3_agent_circular_dispute():
    """Verifies deadlock attribution when 3 distinct agents participate in a circular conflict."""
    engine = DeadlockAttributionEngine()

    entries = [
        BlackboardEntry(
            entry_id="e1",
            parent_id=None,
            branch_id="main",
            agent_id="agent_A",
            agent_role="Architect",
            tag=PXPTag.PROPOSE,
            prediction="Architecture: Microservices",
            explanation="Decoupled services maximize independent deployability.",
            evidence_refs=["scalability", "team_independence"],
            step_number=1,
        ),
        BlackboardEntry(
            entry_id="e2",
            parent_id="e1",
            branch_id="main",
            agent_id="agent_B",
            agent_role="SRE",
            tag=PXPTag.REFUTE,
            prediction="Architecture: Monolith",
            explanation="Operational overhead and network latency of microservices unacceptable.",
            evidence_refs=["operational_complexity"],
            step_number=2,
        ),
        BlackboardEntry(
            entry_id="e3",
            parent_id="e2",
            branch_id="main",
            agent_id="agent_C",
            agent_role="Security",
            tag=PXPTag.REFUTE,
            prediction="Architecture: Serverless",
            explanation="Zero server maintenance and automated least-privilege IAM isolation.",
            evidence_refs=["iam_isolation", "zero_patching"],
            step_number=3,
        ),
        BlackboardEntry(
            entry_id="e4",
            parent_id="e3",
            branch_id="main",
            agent_id="agent_A",
            agent_role="Architect",
            tag=PXPTag.REJECT,
            prediction="Architecture: Microservices",
            explanation="Reject Serverless vendor lock-in and Monolith scaling bottlenecks.",
            evidence_refs=["vendor_lockin"],
            step_number=4,
        ),
        BlackboardEntry(
            entry_id="e5",
            parent_id="e4",
            branch_id="main",
            agent_id="agent_B",
            agent_role="SRE",
            tag=PXPTag.REJECT,
            prediction="Architecture: Monolith",
            explanation="Reject Microservices distributed tracing nightmare and Serverless cold starts.",
            evidence_refs=["cold_starts"],
            step_number=5,
        ),
        BlackboardEntry(
            entry_id="e6",
            parent_id="e5",
            branch_id="main",
            agent_id="agent_C",
            agent_role="Security",
            tag=PXPTag.REJECT,
            prediction="Architecture: Serverless",
            explanation="Reject Monolith blast radius and Microservices network perimeter sprawl.",
            evidence_refs=["blast_radius"],
            step_number=6,
        ),
    ]

    assert engine.is_deadlocked(entries, window=2) is True
    assert engine.is_deadlocked(entries, window=3) is True

    result = engine.analyze(entries)
    assert result.deadlock_detected is True
    assert set(result.conflicting_agents) == {"agent_A", "agent_B", "agent_C"}
    assert len(result.conflict_loop_entries) >= 3
    assert result.divergence_entry_id == "e1"
    assert result.rollback_step == 1
    assert result.contested_predictions["agent_A"] == "Architecture: Microservices"
    assert result.contested_predictions["agent_B"] == "Architecture: Monolith"
    assert result.contested_predictions["agent_C"] == "Architecture: Serverless"
    assert "cold_starts" in result.clashing_evidence
    assert "blast_radius" in result.clashing_evidence
    assert result.attribution_confidence > 0.0


def test_handles_empty_board():
    """Verifies that empty board inputs return clean non-deadlock diagnostics without error."""
    engine = DeadlockAttributionEngine()
    assert engine.is_deadlocked([]) is False
    assert engine.find_consensus_checkpoints([]) == []
    assert engine.locate_divergence_node([]) == (None, 0, [])

    result = engine.analyze([])
    assert result.deadlock_detected is False
    assert "No entries found" in result.reason


def test_handles_single_entry():
    """Verifies that a single entry cannot be considered a deadlock."""
    engine = DeadlockAttributionEngine()
    entries = [
        BlackboardEntry(
            entry_id="e1",
            agent_id="agent_1",
            tag=PXPTag.PROPOSE,
            prediction="Solo Proposal",
            explanation="Initial thesis.",
            step_number=1,
        )
    ]
    assert engine.is_deadlocked(entries) is False
    result = engine.analyze(entries)
    assert result.deadlock_detected is False


def test_no_false_positive_on_single_agent_repetition():
    """Verifies that one agent repeatedly refuting itself is not classified as multi-agent deadlock."""
    engine = DeadlockAttributionEngine()
    entries = [
        BlackboardEntry(
            entry_id="e1",
            agent_id="agent_1",
            tag=PXPTag.REFUTE,
            prediction="Self critique 1",
            explanation="Refuting own prior thought.",
            step_number=1,
        ),
        BlackboardEntry(
            entry_id="e2",
            parent_id="e1",
            agent_id="agent_1",
            tag=PXPTag.REFUTE,
            prediction="Self critique 2",
            explanation="Still self-critiquing.",
            step_number=2,
        ),
    ]
    # Window=2 with only 1 agent should NOT be a deadlock
    assert engine.is_deadlocked(entries, window=2) is False
    result = engine.analyze(entries, window=2)
    assert result.deadlock_detected is False


def test_handles_cyclic_parent_links_gracefully():
    """Verifies that cyclic parent_id links terminate cleanly without infinite loops."""
    engine = DeadlockAttributionEngine()
    # Cyclic entries: e2 points to e3, and e3 points to e2
    entries = [
        BlackboardEntry(
            entry_id="e1",
            parent_id=None,
            agent_id="agent_A",
            tag=PXPTag.PROPOSE,
            prediction="Root",
            explanation="Root proposal.",
            step_number=1,
        ),
        BlackboardEntry(
            entry_id="e2",
            parent_id="e3",  # Cyclic reference to e3
            agent_id="agent_B",
            tag=PXPTag.REFUTE,
            prediction="Cycle B",
            explanation="Disputing C.",
            step_number=2,
        ),
        BlackboardEntry(
            entry_id="e3",
            parent_id="e2",  # Cyclic reference back to e2
            agent_id="agent_C",
            tag=PXPTag.REJECT,
            prediction="Cycle C",
            explanation="Disputing B.",
            step_number=3,
        ),
    ]

    div_id, rollback_step, climb_path = engine.locate_divergence_node(entries)
    # Must terminate without infinite loop and return visited node
    assert div_id in ["e2", "e3"]
    assert len(climb_path) >= 2


def test_handles_unlinked_parent():
    """Verifies that an unlinked/missing parent_id reference terminates cleanly."""
    engine = DeadlockAttributionEngine()
    entries = [
        BlackboardEntry(
            entry_id="e2",
            parent_id="non_existent_entry_id",
            agent_id="agent_A",
            tag=PXPTag.REFUTE,
            prediction="Ghost parent",
            explanation="Missing parent reference.",
            step_number=2,
        ),
        BlackboardEntry(
            entry_id="e3",
            parent_id="e2",
            agent_id="agent_B",
            tag=PXPTag.REJECT,
            prediction="Rejecting ghost",
            explanation="Rejection.",
            step_number=3,
        ),
    ]

    div_id, rollback_step, climb_path = engine.locate_divergence_node(entries)
    assert div_id == "non_existent_entry_id"
    assert rollback_step == 0


def test_branch_filtering_in_analyze():
    """Verifies that analyze filters entries by branch_id so sandbox entries do not contaminate main."""
    engine = DeadlockAttributionEngine()
    entries = [
        BlackboardEntry(
            entry_id="e1",
            branch_id="main",
            agent_id="agent_A",
            tag=PXPTag.PROPOSE,
            prediction="Main diagnosis",
            explanation="Main explanation.",
            step_number=1,
        ),
        BlackboardEntry(
            entry_id="e2",
            branch_id="main",
            agent_id="agent_B",
            tag=PXPTag.RATIFY,
            prediction="Main diagnosis",
            explanation="Main ratify.",
            step_number=2,
        ),
        # Unrelated sandbox branch with conflict entries
        BlackboardEntry(
            entry_id="sb_1",
            branch_id="sandbox_branch",
            agent_id="agent_X",
            tag=PXPTag.REFUTE,
            prediction="Sandbox conflict",
            explanation="Dispute.",
            step_number=3,
        ),
        BlackboardEntry(
            entry_id="sb_2",
            branch_id="sandbox_branch",
            agent_id="agent_Y",
            tag=PXPTag.REJECT,
            prediction="Sandbox conflict",
            explanation="Dispute.",
            step_number=4,
        ),
    ]

    # Analyzing main branch should be healthy (no deadlock)
    result_main = engine.analyze(entries, branch_id="main")
    assert result_main.deadlock_detected is False

    # Analyzing sandbox branch directly should detect the sandbox deadlock
    result_sandbox = engine.analyze(entries, branch_id="sandbox_branch")
    assert result_sandbox.deadlock_detected is True
    assert set(result_sandbox.conflicting_agents) == {"agent_X", "agent_Y"}
