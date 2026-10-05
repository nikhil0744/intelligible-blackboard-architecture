"""
tests/test_sandbox_manager.py
Comprehensive unit tests for SandboxSession and SandboxManager.
"""

import json
from importlib.resources import files
import pytest

from contracts.schemas import BlackboardEntry, PXPTag
from blackboard.store import InMemoryBlackboard
from counterfactual.attribution import AttributionResult, DeadlockAttributionEngine
from sandbox.manager import SandboxManager, SandboxSession


@pytest.fixture
def manager():
    mgr = SandboxManager()
    yield mgr
    mgr.cleanup_all()


@pytest.fixture
def live_debate():
    board = InMemoryBlackboard(
        session_id="session_live_100",
        task_id="task_med_01",
        problem_statement="Patient with atypical dyspnea and clubbing.",
        ground_truth="IPF",
    )
    # Step 1: Initial proposal
    board.add_entry(
        BlackboardEntry(
            entry_id="e1",
            parent_id=None,
            branch_id="main",
            agent_id="agent_cardio",
            agent_role="Cardiologist",
            tag=PXPTag.PROPOSE,
            prediction="CHF",
            explanation="Volume overload suspected.",
            step_number=1,
        )
    )
    # Step 2: Consensus Checkpoint CN
    board.add_entry(
        BlackboardEntry(
            entry_id="e2",
            parent_id="e1",
            branch_id="main",
            agent_id="agent_pulmo",
            agent_role="Pulmonologist",
            tag=PXPTag.RATIFY,
            prediction="CHF",
            explanation="Initial ratify for baseline workup.",
            step_number=2,
            metadata={"is_checkpoint": True},
        )
    )
    # Step 3: Divergence Node
    board.add_entry(
        BlackboardEntry(
            entry_id="e3",
            parent_id="e2",
            branch_id="main",
            agent_id="agent_cardio",
            agent_role="Cardiologist",
            tag=PXPTag.PROPOSE,
            prediction="Diuretic Therapy Escalation",
            explanation="Escalate furosemide dosing.",
            step_number=3,
        )
    )
    # Step 4: Refutation (Deadlock start)
    board.add_entry(
        BlackboardEntry(
            entry_id="e4",
            parent_id="e3",
            branch_id="main",
            agent_id="agent_pulmo",
            agent_role="Pulmonologist",
            tag=PXPTag.REFUTE,
            prediction="IPF Antifibrotic",
            explanation="HRCT confirms honeycombing. Diuretics contraindicated.",
            step_number=4,
        )
    )
    # Step 5: Rejection (Deadlock loop)
    board.add_entry(
        BlackboardEntry(
            entry_id="e5",
            parent_id="e4",
            branch_id="main",
            agent_id="agent_cardio",
            agent_role="Cardiologist",
            tag=PXPTag.REJECT,
            prediction="CHF Congestion",
            explanation="Refusing IPF diagnosis without repeat echocardiogram.",
            step_number=5,
        )
    )
    return board


def test_sandbox_isolation(manager, live_debate):
    """Assert mutations in sandbox do not affect the live blackboard."""
    engine = DeadlockAttributionEngine()
    attribution = engine.analyze(live_debate)

    session = manager.create_session_from_attribution(live_debate, attribution)

    # Add a simulated entry in the sandbox
    sim_entry = BlackboardEntry(
        entry_id="sim_1",
        parent_id="e3",
        branch_id="placeholder",
        agent_id="agent_counterfactual",
        agent_role="Mediator",
        tag=PXPTag.REVISE,
        prediction="Combined Pulmonary-Cardiac Evaluation",
        explanation="Perform high-resolution CT and BNP simultaneously.",
        step_number=6,
    )
    added = session.add_simulated_entry(sim_entry)

    # 1. Assert added entry branch_id was rewritten to sandbox branch
    assert added.branch_id == session.sandbox_branch_id
    assert added in session.get_simulated_entries()

    # 2. Assert live board did not receive the entry
    live_entries = live_debate.get_entries("main")
    live_ids = [e.entry_id for e in live_entries]
    assert "sim_1" not in live_ids
    assert len(live_entries) == 5

    # 3. Assert live board's active branches do not contain the sandbox branch
    live_branches = live_debate.get_state().active_branches
    assert session.sandbox_branch_id not in live_branches


def test_scoped_window_slicing(manager, live_debate):
    """Assert only entries between checkpoint and rollback step are present in sandbox history."""
    engine = DeadlockAttributionEngine()
    attribution = engine.analyze(live_debate)

    assert attribution.latest_checkpoint_step == 2
    assert attribution.rollback_step == 3

    session = manager.create_session_from_attribution(live_debate, attribution)

    all_sandbox_entries = session.get_all_entries()
    sandbox_step_numbers = [e.step_number for e in all_sandbox_entries]
    sandbox_entry_ids = [e.entry_id for e in all_sandbox_entries]

    # Only step 2 (checkpoint) and step 3 (divergence trigger) should be present
    assert sandbox_step_numbers == [2, 3]
    assert sandbox_entry_ids == ["e2", "e3"]

    # Excluded steps
    assert 1 not in sandbox_step_numbers  # Before checkpoint
    assert 4 not in sandbox_step_numbers  # Conflict loop
    assert 5 not in sandbox_step_numbers  # Conflict loop

    # All entries in sandbox board have sandbox_branch_id
    assert all(e.branch_id == session.sandbox_branch_id for e in all_sandbox_entries)
    assert session.start_step == 2
    assert session.rollback_step == 3


def test_3_checkpoint_escalation(manager):
    """Test escalating from CN -> CN-1 -> CN-2, and verify that attempting a 4th jump sets escalation_exhausted=True and returns False."""
    board = InMemoryBlackboard(
        session_id="multi_checkpoint_session",
        task_id="complex_task",
        problem_statement="Multi-phase medical differential.",
    )
    # Checkpoint 1 (CN-2)
    board.add_entry(
        BlackboardEntry(
            entry_id="cp1",
            parent_id=None,
            branch_id="main",
            agent_id="agent_1",
            tag=PXPTag.RATIFY,
            prediction="Initial Consensus",
            explanation="Agreed baseline.",
            step_number=2,
            metadata={"is_checkpoint": True},
        )
    )
    # Checkpoint 2 (CN-1)
    board.add_entry(
        BlackboardEntry(
            entry_id="cp2",
            parent_id="cp1",
            branch_id="main",
            agent_id="agent_2",
            tag=PXPTag.RATIFY,
            prediction="Intermediate Consensus",
            explanation="Agreed secondary workup.",
            step_number=4,
            metadata={"is_checkpoint": True},
        )
    )
    # Checkpoint 3 (CN)
    board.add_entry(
        BlackboardEntry(
            entry_id="cp3",
            parent_id="cp2",
            branch_id="main",
            agent_id="agent_1",
            tag=PXPTag.RATIFY,
            prediction="Latest Consensus",
            explanation="Agreed advanced stage.",
            step_number=6,
            metadata={"is_checkpoint": True},
        )
    )
    # Divergence node
    board.add_entry(
        BlackboardEntry(
            entry_id="div",
            parent_id="cp3",
            branch_id="main",
            agent_id="agent_1",
            tag=PXPTag.PROPOSE,
            prediction="Controversial intervention",
            explanation="Unproven therapy.",
            step_number=7,
        )
    )
    # Deadlock loop
    board.add_entry(
        BlackboardEntry(
            entry_id="ref1",
            parent_id="div",
            branch_id="main",
            agent_id="agent_2",
            tag=PXPTag.REFUTE,
            prediction="Alternative intervention",
            explanation="Refuting intervention.",
            step_number=8,
        )
    )
    board.add_entry(
        BlackboardEntry(
            entry_id="rej1",
            parent_id="ref1",
            branch_id="main",
            agent_id="agent_1",
            tag=PXPTag.REJECT,
            prediction="Controversial intervention",
            explanation="Rejecting refutation.",
            step_number=9,
        )
    )

    engine = DeadlockAttributionEngine()
    attribution = engine.analyze(board)

    session = manager.create_session_from_attribution(board, attribution)

    # Initial state: Level 0 (CN)
    assert len(session.checkpoints_stack) == 3
    assert session.checkpoints_stack[0].entry_id == "cp3"  # CN
    assert session.checkpoints_stack[1].entry_id == "cp2"  # CN-1
    assert session.checkpoints_stack[2].entry_id == "cp1"  # CN-2

    assert session.current_checkpoint_level == 0
    assert session.start_step == 6
    assert session.escalation_exhausted is False

    # Add a simulated entry at level 0
    session.add_simulated_entry(
        BlackboardEntry(
            entry_id="sim_lvl0",
            parent_id="div",
            branch_id="temp",
            agent_id="agent_1",
            tag=PXPTag.REVISE,
            prediction="Compromise 0",
            explanation="Initial compromise trial.",
            step_number=8,
        )
    )
    assert len(session.get_simulated_entries()) == 1

    # Jump 1: CN -> CN-1 (level 1)
    jump1 = session.escalate_to_prior_checkpoint()
    assert jump1 is True
    assert session.current_checkpoint_level == 1
    assert session.start_step == 4
    assert session.escalation_exhausted is False

    # Verify board history was reloaded to include cp2 and stale simulated entry was purged
    assert len(session.get_simulated_entries()) == 0
    reloaded_ids_lvl1 = [e.entry_id for e in session.get_all_entries()]
    assert "cp2" in reloaded_ids_lvl1
    assert "cp3" in reloaded_ids_lvl1
    assert "sim_lvl0" not in reloaded_ids_lvl1

    # Jump 2: CN-1 -> CN-2 (level 2)
    jump2 = session.escalate_to_prior_checkpoint()
    assert jump2 is True
    assert session.current_checkpoint_level == 2
    assert session.start_step == 2
    assert session.escalation_exhausted is False

    # Verify board history was reloaded to include cp1
    reloaded_ids_lvl2 = [e.entry_id for e in session.get_all_entries()]
    assert "cp1" in reloaded_ids_lvl2
    assert "cp2" in reloaded_ids_lvl2

    # Jump 3 attempt: beyond level 2 (level >= 3), triggers exhaustion
    jump3 = session.escalate_to_prior_checkpoint()
    assert jump3 is False
    assert session.escalation_exhausted is True

    # Subsequent attempts continue to return False
    jump4 = session.escalate_to_prior_checkpoint()
    assert jump4 is False
    assert session.escalation_exhausted is True


def test_graph_telemetry_export(manager, live_debate):
    """Test output structure of export_graph_for_telemetry (nodes, edges, is_simulated flags, no dangling edges)."""
    engine = DeadlockAttributionEngine()
    attribution = engine.analyze(live_debate)

    session = manager.create_session_from_attribution(live_debate, attribution)

    # Add simulated entry
    session.add_simulated_entry(
        BlackboardEntry(
            entry_id="sim_node_1",
            parent_id="e3",
            branch_id="placeholder",
            agent_id="agent_mediator",
            agent_role="Mediator",
            tag=PXPTag.REVISE,
            prediction="Unified Plan",
            explanation="Reconciling cardio and pulmo data.",
            step_number=4,
        )
    )

    telemetry = session.export_graph_for_telemetry()

    assert "nodes" in telemetry
    assert "edges" in telemetry
    assert telemetry["session_id"] == session.session_id
    assert telemetry["sandbox_branch_id"] == session.sandbox_branch_id

    nodes = telemetry["nodes"]
    edges = telemetry["edges"]

    # Validate node attributes and position
    for node in nodes:
        assert "id" in node
        assert "label" in node
        assert "tag" in node
        assert "agent" in node
        assert "parent_id" in node
        assert "is_simulated" in node
        assert "position" in node
        assert node["position"] == {"x": 0.0, "y": 0.0}

    # Check simulated flags
    node_map = {n["id"]: n for n in nodes}
    assert node_map["e2"]["is_simulated"] is False
    assert node_map["e3"]["is_simulated"] is False
    assert node_map["sim_node_1"]["is_simulated"] is True

    # Validate edges: No dangling edge for e2 -> e1 (since e1 is outside scoped window)
    edge_map = {e["id"]: e for e in edges}
    assert "edge_e1_e2" not in edge_map
    assert "edge_e2_e3" in edge_map
    assert edge_map["edge_e2_e3"]["is_simulated"] is False
    assert "edge_e3_sim_node_1" in edge_map
    assert edge_map["edge_e3_sim_node_1"]["is_simulated"] is True

    # Ensure every edge's source and target exist in nodes
    all_node_ids = set(node_map.keys())
    for e in edges:
        assert e["source"] in all_node_ids
        assert e["target"] in all_node_ids


def test_session_cleanup(manager, live_debate):
    """Test close_session and cleanup_all, including lookup by sandbox_branch_id."""
    engine = DeadlockAttributionEngine()
    attribution = engine.analyze(live_debate)

    session_1 = manager.create_session_from_attribution(live_debate, attribution)
    session_2 = manager.create_session_from_attribution(live_debate, attribution)

    # Lookup by session_id
    assert manager.get_session(session_1.session_id) is not None
    assert manager.get_session(session_2.session_id) is not None

    # Lookup by sandbox_branch_id
    assert manager.get_session(session_1.sandbox_branch_id) is session_1
    assert manager.get_session(session_2.sandbox_branch_id) is session_2

    # Close session 1 by session_id
    closed = manager.close_session(session_1.session_id)
    assert closed is True
    assert manager.get_session(session_1.session_id) is None
    assert manager.get_session(session_1.sandbox_branch_id) is None
    assert manager.get_session(session_2.session_id) is not None

    # Close session 2 by sandbox_branch_id
    closed_2 = manager.close_session(session_2.sandbox_branch_id)
    assert closed_2 is True
    assert manager.get_session(session_2.session_id) is None

    # Cleanup all
    session_3 = manager.create_session_from_attribution(live_debate, attribution)
    assert manager.get_session(session_3.session_id) is not None
    manager.cleanup_all()
    assert manager.get_session(session_3.session_id) is None


def test_fixture_sample_deadlock_session(manager):
    """Test creating a sandbox session from sample_deadlocks.json fixture."""
    fixture_path = files("fixtures").joinpath("sample_deadlocks.json")
    with fixture_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    board = InMemoryBlackboard(
        session_id=data["session_id"],
        task_id=data["task_id"],
        problem_statement=data["problem_statement"],
        ground_truth=data["ground_truth"],
    )
    for entry_dict in data["entries"]:
        board.add_entry(BlackboardEntry.model_validate(entry_dict))

    engine = DeadlockAttributionEngine()
    attribution = engine.analyze(board)
    assert attribution.deadlock_detected is True

    session = manager.create_session_from_attribution(board, attribution)
    assert session is not None
    assert session.sandbox_branch_id.startswith("sandbox_cf_")
    assert session.rollback_step == 1

    # In sample_deadlocks.json, there are no consensus checkpoints, so start_step is 1
    assert session.start_step == 1
    all_entries = session.get_all_entries()
    assert len(all_entries) == 1
    assert all_entries[0].entry_id == "e1"


def test_escalate_empty_checkpoints(manager):
    """Assert escalating with empty checkpoints immediately exhausts and returns False."""
    board = InMemoryBlackboard(session_id="s_empty", task_id="t1", problem_statement="P")
    attribution = AttributionResult(
        deadlock_detected=True,
        divergence_entry_id="e1",
        rollback_step=1,
        latest_checkpoint_step=0,
    )
    session = manager.create_session_from_attribution(board, attribution)
    ok = session.escalate_to_prior_checkpoint()
    assert ok is False
    assert session.escalation_exhausted is True


def test_escalate_bounds_clamped(manager):
    """Verify that start_step never exceeds rollback_step even if checkpoint has higher step."""
    board = InMemoryBlackboard(session_id="s_clamp", task_id="t1", problem_statement="P")
    cp = BlackboardEntry(
        entry_id="cp_high",
        parent_id=None,
        branch_id="main",
        agent_id="a1",
        tag=PXPTag.RATIFY,
        prediction="P",
        explanation="E",
        step_number=10,
    )
    session = SandboxSession(
        session_id="s1",
        source_session_id="src1",
        sandbox_branch_id="sandbox_cf_clamp",
        board=board,
        start_step=2,
        rollback_step=5,
        checkpoints_stack=[cp, cp],  # dummy stack with step 10
        current_checkpoint_level=0,
    )

    escalated = session.escalate_to_prior_checkpoint()
    assert escalated is True
    assert session.start_step == 5  # Clamped to rollback_step


def test_add_simulated_entry_mutation_safety(manager, live_debate):
    """Verify that mutating the returned entry from add_simulated_entry does not mutate internal state."""
    engine = DeadlockAttributionEngine()
    attribution = engine.analyze(live_debate)
    session = manager.create_session_from_attribution(live_debate, attribution)

    sim = BlackboardEntry(
        entry_id="sim_mut",
        parent_id="e3",
        branch_id="temp",
        agent_id="agent_1",
        tag=PXPTag.REVISE,
        prediction="Original Prediction",
        explanation="Original Explanation",
        step_number=4,
    )
    returned = session.add_simulated_entry(sim)
    # Mutate the returned object
    returned.prediction = "MUTATED_PREDICTION"

    # Internal state must still have original
    stored = session.get_simulated_entries()[0]
    assert stored.prediction == "Original Prediction"
