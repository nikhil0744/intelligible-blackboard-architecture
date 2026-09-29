"""
tests/test_deadlock_solver.py
Comprehensive integration tests for Task 5: Live Deadlock Solver Daemon & Preemption Bridge.
Verifies automated deadlock detection, retrospective sandbox recovery,
Priority 0 preemption injection, and live debate resolution into CONSENSUS.
"""

from typing import Any, Dict, List
import pytest

from contracts.schemas import (
    AgentContribution,
    AgentRole,
    BlackboardEntry,
    BlackboardSnapshot,
    BlackboardStatus,
    DeadlockNotification,
    Explanation,
    PEXPayload,
    Prediction,
    PXPTag,
    ReviseProposal,
    TelemetryEventType,
)
from counterfactual.solver import DeadlockSolverDaemon, SolverConfig
from fixtures.mocks import MockLLMBroker, MockPEXAgent
from scheduler.engine import DeterministicScheduler
from scheduler.policies import RoundRobinPolicy
from storage.redis_store import InMemoryRedisClient, RedisBlackboardStore


@pytest.fixture
def redis_store() -> RedisBlackboardStore:
    client = InMemoryRedisClient()
    return RedisBlackboardStore(redis_client=client)


def test_solver_daemon_lifecycle(redis_store: RedisBlackboardStore):
    session_id = "sess_lifecycle_001"
    redis_store.create_session(session_id, "task_life", "Lifecycle test")
    scheduler = DeterministicScheduler(session_id=session_id, store=redis_store)

    config = SolverConfig(convergence_threshold=0.75, mediator_agent_id="test_mediator")
    daemon = DeadlockSolverDaemon(config=config)

    assert daemon.config.convergence_threshold == 0.75
    assert daemon.config.mediator_agent_id == "test_mediator"
    assert len(daemon.get_recovery_history()) == 0

    # Attach to scheduler
    daemon.attach_to_scheduler(scheduler)
    assert daemon in scheduler._deadlock_hooks

    # Detach
    daemon.detach()
    assert daemon not in scheduler._deadlock_hooks


def test_end_to_end_deadlock_recovery_with_scheduler(redis_store: RedisBlackboardStore):
    """
    End-to-end integration test:
    1. Agent 1 and Agent 2 argue on the live blackboard (REFUTE vs REJECT).
    2. Scheduler detects deadlock and triggers DeadlockSolverDaemon.
    3. Daemon simulates resolution in sandbox, evaluates with scoring engine,
       and returns a Priority 0 ReviseProposal.
    4. Scheduler injects proposal, unpauses queue, and executes recovery turn.
    5. Disputing agents accept the synthesis and ratify, leading to CONSENSUS.
    """
    session_id = "sess_e2e_recovery_001"
    redis_store.create_session(
        session_id=session_id,
        task_id="task_e2e",
        task_description="Differentiate clinical criteria for pulmonary disease.",
    )

    scheduler = DeterministicScheduler(
        session_id=session_id,
        store=redis_store,
        policy=RoundRobinPolicy(),
        deadlock_window=2,
        consensus_threshold=2,
    )

    # Brokers for disputing agents
    broker_a = MockLLMBroker(default_response={
        "tag": "REFUTE",
        "prediction": "Diagnosis A (IPF)",
        "explanation": "Honeycombing on CT supports IPF.",
        "confidence": 0.85,
        "evidence_refs": ["ct_honeycombing"],
    })
    broker_b = MockLLMBroker(default_response={
        "tag": "REJECT",
        "prediction": "Diagnosis B (SSc-ILD)",
        "explanation": "Positive ANA and sclerodactyly support CTD-ILD.",
        "confidence": 0.90,
        "evidence_refs": ["ana_positive"],
    })

    agent_a = MockPEXAgent(agent_id="doc_pulmonology", agent_role="primary", broker=broker_a)
    agent_b = MockPEXAgent(agent_id="doc_rheumatology", agent_role="critic", broker=broker_b)

    scheduler.register_agent("doc_pulmonology", AgentRole.PRIMARY, agent_a)
    scheduler.register_agent("doc_rheumatology", AgentRole.CRITIC, agent_b)

    # Configure solver daemon with broker that provides consensus compromise
    solver_broker = MockLLMBroker(default_response={
        "tag": "RATIFY",
        "prediction": "Unified Assessment: Connective Tissue Disease-Associated ILD",
        "explanation": "CT honeycombing combined with positive ANA represents collagen vascular ILD.",
        "confidence": 0.95,
        "evidence_refs": ["ct_honeycombing", "ana_positive"],
    })

    daemon = DeadlockSolverDaemon(
        config=SolverConfig(
            convergence_threshold=0.70,
            mediator_agent_id="counterfactual_resolver",
            mediator_role=AgentRole.COUNTERFACTUAL,
        ),
        broker=solver_broker,
    )
    daemon.attach_to_scheduler(scheduler)

    # Step 1: Agent A refutes
    c1 = scheduler.step()
    assert c1.tag == PXPTag.REFUTE
    assert scheduler.queue.is_paused is False

    # Step 2: Agent B rejects -> Deadlock window of 2 triggers!
    # Daemon automatically executes in hook and injects Priority 0 proposal!
    c2 = scheduler.step()
    assert c2.tag == PXPTag.REJECT
    assert scheduler.queue.is_paused is False  # Resumed by daemon injection!

    # Verify recovery history in daemon
    history = daemon.get_recovery_history()
    assert len(history) == 1
    recovery = history[0]
    assert recovery["convergence_score"] >= 0.70
    assert "counterfactual_resolver" in recovery["proposal"].revised_contribution.agent_id

    # Verify Priority 0 preemption turn is at the top of the queue
    assert scheduler.queue.peek().priority == 0
    assert scheduler.queue.peek().agent_id == "counterfactual_resolver"

    # Step 3: Priority 0 counterfactual resolution turn executes!
    c3 = scheduler.step()
    assert c3.agent_id == "counterfactual_resolver"
    assert c3.tag == PXPTag.REVISE
    assert c3.is_counterfactual is True
    assert "Unified Assessment" in c3.prediction

    # Now both doctors respond to the compromise with RATIFY
    broker_a.default_response = {
        "tag": "RATIFY",
        "prediction": "Unified Assessment: Connective Tissue Disease-Associated ILD",
        "explanation": "Agreed, unifying criteria is sound.",
        "confidence": 0.92,
        "evidence_refs": ["ct_honeycombing", "ana_positive"],
    }
    broker_b.default_response = {
        "tag": "RATIFY",
        "prediction": "Unified Assessment: Connective Tissue Disease-Associated ILD",
        "explanation": "Concur with unified diagnostic framework.",
        "confidence": 0.94,
        "evidence_refs": ["ct_honeycombing", "ana_positive"],
    }

    # Step 4: Agent A ratifies
    c4 = scheduler.step()
    assert c4.tag == PXPTag.RATIFY

    # Step 5: Agent B ratifies -> Consensus threshold of 2 met!
    c5 = scheduler.step()
    assert c5.tag == PXPTag.RATIFY

    # Verify final blackboard status reached CONSENSUS
    snap = redis_store.get_snapshot(session_id)
    assert snap.status == BlackboardStatus.CONSENSUS
    assert len(snap.contributions) == 5


def test_solver_escalates_on_suboptimal_branch(redis_store: RedisBlackboardStore):
    """
    Verifies that when Tier 1 simulation does not reach the convergence threshold,
    the daemon backtracks and escalates to prior consensus checkpoints.
    """
    session_id = "sess_escalation_001"
    redis_store.create_session(session_id, "task_esc", "Escalation test")

    # Add baseline consensus checkpoint at step 1
    cp0 = BlackboardEntry(
        entry_id="c_cp_0",
        session_id=session_id,
        agent_id="agent_init",
        tag=PXPTag.RATIFY,
        prediction="Initial Patient Baseline",
        explanation="Baseline clinical findings",
        step_number=1,
    )
    # Add subsequent consensus checkpoint at step 2
    cp1 = BlackboardEntry(
        entry_id="c_cp_1",
        parent_id="c_cp_0",
        session_id=session_id,
        agent_id="agent_init",
        tag=PXPTag.RATIFY,
        prediction="Updated Diagnostic Hypothesis",
        explanation="Preliminary consensus criteria",
        step_number=2,
    )

    # Deadlock window (turns 3 and 4)
    d1 = BlackboardEntry(
        entry_id="c_d_1",
        parent_id="c_cp_1",
        session_id=session_id,
        agent_id="agent_a",
        tag=PXPTag.REFUTE,
        prediction="Option A",
        explanation="Defending option A",
        step_number=3,
    )
    d2 = BlackboardEntry(
        entry_id="c_d_2",
        parent_id="c_d_1",
        session_id=session_id,
        agent_id="agent_b",
        tag=PXPTag.REJECT,
        prediction="Option B",
        explanation="Rejecting option A",
        step_number=4,
    )
    snap = redis_store.get_snapshot(session_id)
    snap.contributions = [
        cp0.to_agent_contribution(session_id=session_id),
        cp1.to_agent_contribution(session_id=session_id),
        d1.to_agent_contribution(session_id=session_id),
        d2.to_agent_contribution(session_id=session_id),
    ]
    snap.version = 4
    redis_store._save_snapshot(snap)

    notif = DeadlockNotification(
        session_id=session_id,
        deadlock_turn_index=4,
        conflicting_agent_ids=["agent_a", "agent_b"],
        repeated_tags=[PXPTag.REFUTE, PXPTag.REJECT],
        snapshot=redis_store.get_snapshot(session_id),
        reason="Disagreement on treatment selection",
    )

    # Broker that returns poor response first (causing escalation), then good response
    tiered_broker = MockLLMBroker(
        default_response={"tag": "REFUTE", "prediction": "Default failure", "explanation": "Failed attempt", "confidence": 0.5},
        responses=[
            # Level 0 attempt: low score REFUTE (iter 1 & iter 2)
            {"tag": "REFUTE", "prediction": "Weak revision 1", "explanation": "Weak explanation", "confidence": 0.5},
            {"tag": "REFUTE", "prediction": "Still opposing 1", "explanation": "Still opposing", "confidence": 0.5},
            {"tag": "REFUTE", "prediction": "Weak revision 2", "explanation": "Weak explanation", "confidence": 0.5},
            {"tag": "REFUTE", "prediction": "Still opposing 2", "explanation": "Still opposing", "confidence": 0.5},
            # Level 1 attempt: high score RATIFY (with reasoning depth and mutual evidence)
            {
                "tag": "RATIFY",
                "prediction": "Strong consensus",
                "explanation": "Comprehensive clinical synthesis confirms diagnosis because mutual evidence demonstrates full alignment.",
                "evidence_refs": ["ev_clin_1", "ev_clin_2"],
                "confidence": 0.95,
            },
            {
                "tag": "RATIFY",
                "prediction": "Strong consensus",
                "explanation": "Comprehensive clinical synthesis confirms diagnosis because mutual evidence demonstrates full alignment.",
                "evidence_refs": ["ev_clin_1", "ev_clin_2"],
                "confidence": 0.95,
            },
        ]
    )

    daemon = DeadlockSolverDaemon(
        config=SolverConfig(convergence_threshold=0.70, max_simulation_rounds=2),
        broker=tiered_broker,
    )

    proposal = daemon.resolve(notif)
    assert proposal is not None
    assert proposal.revised_contribution.tag == PXPTag.REVISE
    assert "Strong consensus" in proposal.revised_contribution.prediction


def test_solver_handles_unresolvable_deadlock_gracefully(redis_store: RedisBlackboardStore):
    """
    Verifies that if all simulation attempts fail across all escalation levels,
    the daemon gracefully returns None without crashing.
    """
    session_id = "sess_unresolvable_001"
    redis_store.create_session(session_id, "task_unres", "Unresolvable test")

    d1 = BlackboardEntry(
        entry_id="c_1",
        session_id=session_id,
        agent_id="a1",
        tag=PXPTag.REFUTE,
        prediction="P1",
        explanation="E1",
        step_number=1,
    )
    d2 = BlackboardEntry(
        entry_id="c_2",
        parent_id="c_1",
        session_id=session_id,
        agent_id="a2",
        tag=PXPTag.REJECT,
        prediction="P2",
        explanation="E2",
        step_number=2,
    )
    snap = redis_store.get_snapshot(session_id)
    snap.contributions = [
        d1.to_agent_contribution(session_id=session_id),
        d2.to_agent_contribution(session_id=session_id),
    ]
    snap.version = 2
    redis_store._save_snapshot(snap)

    notif = DeadlockNotification(
        session_id=session_id,
        deadlock_turn_index=2,
        conflicting_agent_ids=["a1", "a2"],
        repeated_tags=[PXPTag.REFUTE, PXPTag.REJECT],
        snapshot=redis_store.get_snapshot(session_id),
        reason="Irreconcilable dispute",
    )

    # Broker that always rejects
    failing_broker = MockLLMBroker(default_response={
        "tag": "REJECT",
        "prediction": "Hostile refusal",
        "explanation": "Will never agree",
        "confidence": 0.99,
    })

    daemon = DeadlockSolverDaemon(
        config=SolverConfig(convergence_threshold=0.85, max_escalation_levels=2),
        broker=failing_broker,
    )

    proposal = daemon.resolve(notif)
    assert proposal is None


def test_telemetry_event_generation(redis_store: RedisBlackboardStore):
    """
    Verifies telemetry events are emitted to the attached scheduler.
    """
    session_id = "sess_telem_001"
    redis_store.create_session(session_id, "task_tel", "Telemetry test")
    scheduler = DeterministicScheduler(session_id=session_id, store=redis_store)

    d1 = BlackboardEntry(entry_id="c_1", session_id=session_id, agent_id="a1", tag=PXPTag.REFUTE, prediction="P1", explanation="E1", step_number=1)
    d2 = BlackboardEntry(entry_id="c_2", parent_id="c_1", session_id=session_id, agent_id="a2", tag=PXPTag.REJECT, prediction="P2", explanation="E2", step_number=2)
    snap_tel = redis_store.get_snapshot(session_id)
    snap_tel.contributions = [
        d1.to_agent_contribution(session_id=session_id),
        d2.to_agent_contribution(session_id=session_id),
    ]
    snap_tel.version = 2
    redis_store._save_snapshot(snap_tel)

    notif = DeadlockNotification(
        session_id=session_id,
        deadlock_turn_index=2,
        conflicting_agent_ids=["a1", "a2"],
        repeated_tags=[PXPTag.REFUTE, PXPTag.REJECT],
        snapshot=redis_store.get_snapshot(session_id),
        reason="Telemetry verification",
    )

    emitted_events: List[Any] = []
    scheduler._emit_telemetry = lambda event_type, payload: emitted_events.append((event_type, payload))

    daemon = DeadlockSolverDaemon(config=SolverConfig(telemetry_enabled=True))
    daemon.attach_to_scheduler(scheduler)

    proposal = daemon.resolve(notif)
    assert proposal is not None

    # Check telemetry
    assert len(emitted_events) == 1
    event_type, payload = emitted_events[0]
    assert event_type == TelemetryEventType.COUNTERFACTUAL_RESOLVED
    assert payload["session_id"] == session_id
    assert payload["convergence_score"] >= 0.70
    assert payload["mediator_agent_id"] == "counterfactual_resolver"
