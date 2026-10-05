"""
fixtures/spike_starter.py
Starter script for Student 3's Week 1 Research Spike.
Demonstrates:
1. Loading the sample deadlock debate
2. Identifying the conflict attribution point
3. Cloning the blackboard into an isolated sandbox
4. Generating an alternative counterfactual turn via a mock broker
5. Re-injecting a winning REVISE resolution back into the live debate
"""

import json
from importlib.resources import files
from contracts.schemas import BlackboardEntry, PXPTag
from blackboard.store import InMemoryBlackboard
from fixtures.mocks import MockLLMBroker, MockPEXAgent


def run_spike():
    print("=" * 60)
    print("STUDENT 3: COUNTERFACTUAL SPIKE PROTOTYPE")
    print("=" * 60)

    # 1. Load the recorded deadlock scenario
    fixture_file = files("fixtures").joinpath("sample_deadlocks.json")
    with fixture_file.open("r") as f:
        data = json.load(f)

    board = InMemoryBlackboard(
        session_id=data["session_id"],
        task_id=data["task_id"],
        problem_statement=data["problem_statement"],
        ground_truth=data["ground_truth"],
    )

    for entry_dict in data["entries"]:
        board.add_entry(BlackboardEntry(**entry_dict))

    print(f"\n[1] Loaded blackboard session: '{board.get_state().session_id}'")
    print(f"Problem: {board.get_state().problem_statement[:80]}...")
    print(f"Ground truth target: {board.get_state().ground_truth}")

    entries = board.get_entries("main")
    print(f"\nCurrent entries on main branch ({len(entries)} turns):")
    for e in entries:
        print(f"  Step {e.step_number} [{e.agent_id} | {e.tag.value}]: {e.prediction}")

    # 2. Check for Deadlock
    is_deadlocked = board.detect_deadlock(window=2)
    print(f"\n[2] Deadlock detected? -> {is_deadlocked}")

    # 3. Attribution (Student 3's core research challenge)
    # The deadlock occurred between e3 and e4, which originated from e2's REFUTE of e1.
    rollback_step = 2
    print(f"\n[3] Rolling back to Step {rollback_step} for sandbox simulation...")

    # 4. Create isolated sandbox
    sandbox = board.clone_sandbox(sandbox_branch_id="sandbox_cf_spike", up_to_step=rollback_step)
    print(f"Sandbox created with {len(sandbox.get_entries('sandbox_cf_spike'))} historical entries.")

    # 5. Simulate 'What-If' alternative turn forward
    broker = MockLLMBroker()
    broker.enqueue_response({
        "tag": PXPTag.REVISE.value,
        "prediction": "Idiopathic Pulmonary Fibrosis (IPF) with secondary cardiac monitoring",
        "explanation": "Acknowledge IPF given pathognomonic honeycombing; revise cardiac focus to monitoring.",
        "confidence": 0.95,
        "evidence_refs": ["honeycombing", "clubbing"],
    })

    cf_agent = MockPEXAgent(agent_id="agent_cardio", agent_role="Cardiologist", broker=broker)
    winning_turn = cf_agent.generate_turn(
        blackboard=sandbox,
        branch_id="sandbox_cf_spike",
        parent_id="e2",
        custom_prompt="What if the Cardiologist conceded on IPF and suggested secondary monitoring?"
    )

    print(f"\n[4] Simulated Counterfactual Turn Generated:")
    print(f"  Tag: {winning_turn.tag.value}")
    print(f"  Prediction: {winning_turn.prediction}")
    print(f"  Explanation: {winning_turn.explanation}")

    # 6. Inject the resolution back to the live board
    winning_turn.branch_id = "main"
    winning_turn.step_number = len(board.get_entries("main")) + 1
    winning_turn.metadata["counterfactual_resolution"] = True
    board.add_entry(winning_turn)

    print(f"\n[5] Resolution injected back to live board! Total turns now: {len(board.get_entries('main'))}")
    print(f"Deadlock cleared? -> {not board.detect_deadlock(window=2)}")
    
    try:
        from streaming.trace_exporter import export_from_spike_blackboard
        export_from_spike_blackboard(board, sandbox, winning_turn, output_path="outputs/live_trace.json")
        print("[+] Exported live execution trace to outputs/live_trace.json")
    except Exception as e:
        print(f"[!] Note: Trace export failed: {e}")
    print("=" * 60)


if __name__ == "__main__":
    run_spike()
