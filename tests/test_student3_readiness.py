"""
tests/test_student3_readiness.py
Validates that all minimum prerequisites for Student 3 are operational:
1. Pydantic contracts & schemas
2. Thread-safe in-memory Blackboard & cloning
3. Deadlock detection heuristic
4. Mock agent & LLM broker
5. Loading sample deadlock fixture & executing a sandbox roll-forward
"""

import json
from importlib.resources import files
import pytest
from contracts.schemas import BlackboardEntry, BlackboardState, PXPTag
from blackboard.store import InMemoryBlackboard
from fixtures.mocks import MockLLMBroker, MockPEXAgent


def test_contracts_schema_validation():
    """Verify that schemas enforce valid PXP tags and non-empty fields."""
    entry = BlackboardEntry(
        entry_id="e1",
        agent_id="agent_1",
        tag=PXPTag.PROPOSE,
        prediction="Diagnosis X",
        explanation="Clinical rationale based on elevated biomarkers.",
        confidence=0.92,
    )
    assert entry.tag == PXPTag.PROPOSE
    assert entry.confidence == 0.92

    # Verification that empty predictions raise validation errors
    with pytest.raises(ValueError):
        BlackboardEntry(
            entry_id="e2",
            agent_id="agent_1",
            tag=PXPTag.PROPOSE,
            prediction="",
            explanation="Some reason",
        )


def test_blackboard_cloning_and_sandbox_branching():
    """Verify that the in-memory store can clone historical slices for Student 3's sandbox."""
    board = InMemoryBlackboard(
        session_id="sess_101",
        task_id="task_1",
        problem_statement="Patient with acute chest pain.",
    )

    board.add_entry(BlackboardEntry(
        entry_id="e1", agent_id="A", tag=PXPTag.PROPOSE,
        prediction="STEMI", explanation="ST elevations on ECG", step_number=1
    ))
    board.add_entry(BlackboardEntry(
        entry_id="e2", parent_id="e1", agent_id="B", tag=PXPTag.REFUTE,
        prediction="Aortic Dissection", explanation="Tearing pain radiating to back", step_number=2
    ))
    board.add_entry(BlackboardEntry(
        entry_id="e3", parent_id="e2", agent_id="A", tag=PXPTag.REJECT,
        prediction="STEMI", explanation="ECG findings take priority", step_number=3
    ))

    # Clone up to step 2 into a sandbox branch
    sandbox = board.clone_sandbox(sandbox_branch_id="sandbox_branch_1", up_to_step=2)
    sandbox_entries = sandbox.get_entries("sandbox_branch_1")

    assert len(sandbox_entries) == 2
    assert sandbox_entries[0].entry_id == "e1"
    assert sandbox_entries[1].entry_id == "e2"
    assert sandbox_entries[1].branch_id == "sandbox_branch_1"


def test_fixture_deadlock_detection_and_resolution():
    """Load sample deadlock fixture, detect deadlock, and simulate counterfactual resolution."""
    fixture_path = files("fixtures").joinpath("sample_deadlocks.json")
    with fixture_path.open("r") as f:
        fixture_data = json.load(f)

    # Initialize blackboard from fixture
    board = InMemoryBlackboard(
        session_id=fixture_data["session_id"],
        task_id=fixture_data["task_id"],
        problem_statement=fixture_data["problem_statement"],
        ground_truth=fixture_data["ground_truth"],
    )

    for entry_dict in fixture_data["entries"]:
        board.add_entry(BlackboardEntry(**entry_dict))

    # Deadlock heuristic check: steps 3 and 4 were REJECT
    assert board.detect_deadlock(window=2) is True

    # Student 3 Sandbox Workflow:
    # 1. Clone history rolled back to step 2 (prior to the cascading REJECT loop)
    sandbox = board.clone_sandbox(sandbox_branch_id="cf_sandbox_med_1", up_to_step=2)

    # 2. Configure a mock broker that generates a counterfactual REVISE synthesis
    broker = MockLLMBroker()
    broker.enqueue_response({
        "tag": PXPTag.REVISE.value,
        "prediction": "Idiopathic Pulmonary Fibrosis (IPF) with secondary cardiac monitoring",
        "explanation": "Accepting HRCT honeycombing as conclusive evidence for IPF; revised cardiac stance to secondary monitoring.",
        "confidence": 0.94,
        "evidence_refs": ["honeycombing", "clubbing"],
    })

    cf_agent = MockPEXAgent(agent_id="agent_cardio", agent_role="Cardiologist", broker=broker)

    # 3. Simulate alternative turn forward in the sandbox
    revised_turn = cf_agent.generate_turn(
        blackboard=sandbox,
        branch_id="cf_sandbox_med_1",
        parent_id="e2",
    )

    assert revised_turn.tag == PXPTag.REVISE
    assert "IPF" in revised_turn.prediction

    # 4. Inject the winning counterfactual back into the live blackboard
    revised_turn.branch_id = "main"
    revised_turn.step_number = 5
    revised_turn.metadata["counterfactual_source"] = "cf_sandbox_med_1"
    board.add_entry(revised_turn)

    # 5. Verify the deadlock is broken
    assert board.detect_deadlock(window=2) is False
    all_main_entries = board.get_entries("main")
    assert len(all_main_entries) == 5
    assert all_main_entries[-1].tag == PXPTag.REVISE
