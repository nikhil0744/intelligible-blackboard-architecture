"""
tests/test_scheduler.py
Comprehensive unit, arbitration, and integration tests for the Deterministic Scheduler.
Validates turn priority queues, round-robin & reactive PXP arbitration policies,
distributed write-lock mediation with RedisBlackboardStore, terminal consensus detection,
deadlock notification hooks, and counterfactual recovery injection.
"""

from datetime import datetime, timedelta, timezone
import pytest

from contracts.schemas import (
    AgentContribution,
    AgentRole,
    BlackboardSnapshot,
    BlackboardStatus,
    DeadlockNotification,
    Explanation,
    PEXPayload,
    Prediction,
    PXPTag,
    ReviseProposal,
    ScheduledTurn,
)
from fixtures.mocks import MockLLMBroker, MockPEXAgent
from scheduler.engine import DeterministicScheduler
from scheduler.policies import PriorityPolicy, ReactivePXPPolicy, RoundRobinPolicy
from scheduler.queue import TurnPriorityQueue
from storage.redis_store import InMemoryRedisClient, RedisBlackboardStore


# ==============================================================================
# 1. TurnPriorityQueue Unit Tests
# ==============================================================================

def test_priority_queue_ordering():
    pq = TurnPriorityQueue(session_id="test_sess")
    now = datetime.now(timezone.utc)

    # Enqueue turns with varying priorities and timestamps
    t_normal = ScheduledTurn(session_id="test_sess", agent_id="agent_mid", priority=2, scheduled_at=now)
    t_cf = ScheduledTurn(session_id="test_sess", agent_id="agent_cf", priority=0, scheduled_at=now + timedelta(seconds=1))
    t_critic = ScheduledTurn(session_id="test_sess", agent_id="agent_critic", priority=1, scheduled_at=now + timedelta(seconds=2))

    pq.enqueue(t_normal)
    pq.enqueue(t_cf)
    pq.enqueue(t_critic)

    assert len(pq) == 3
    assert pq.is_empty() is False

    # Priority 0 must be dequeued first, followed by 1, then 2
    first = pq.dequeue()
    assert first is not None and first.agent_id == "agent_cf" and first.priority == 0

    second = pq.dequeue()
    assert second is not None and second.agent_id == "agent_critic" and second.priority == 1

    third = pq.dequeue()
    assert third is not None and third.agent_id == "agent_mid" and third.priority == 2

    assert pq.dequeue() is None
    assert pq.is_empty() is True


def test_priority_queue_tie_breaking_and_filtering():
    pq = TurnPriorityQueue(session_id="test_sess_2")
    now = datetime.now(timezone.utc)

    t1 = ScheduledTurn(session_id="test_sess_2", agent_id="agent_a", priority=1, scheduled_at=now)
    t2 = ScheduledTurn(session_id="test_sess_2", agent_id="agent_b", priority=1, scheduled_at=now + timedelta(seconds=1))
    t3 = ScheduledTurn(session_id="test_sess_2", agent_id="agent_a", priority=1, scheduled_at=now + timedelta(seconds=2))

    pq.enqueue(t2)
    pq.enqueue(t1)
    pq.enqueue(t3)

    # Tie-breaking by scheduled_at: t1 was scheduled earliest
    assert pq.peek().agent_id == "agent_a"

    # Filter out agent_a
    removed = pq.remove_by_agent("agent_a")
    assert removed == 2
    assert len(pq) == 1
    assert pq.dequeue().agent_id == "agent_b"


# ==============================================================================
# 2. Arbitration Policies Unit Tests
# ==============================================================================

def test_round_robin_policy():
    policy = RoundRobinPolicy()
    agents = {
        "agent_1": {"role": "primary", "priority": 1},
        "agent_2": {"role": "critic", "priority": 1},
        "agent_3": {"role": "arbiter", "priority": 1},
    }
    dummy_snap = BlackboardSnapshot(
        session_id="sess_rr",
        task_id="t1",
        task_description="Task 1",
    )

    t1 = policy.next_turns("sess_rr", dummy_snap, agents)[0]
    assert t1.agent_id == "agent_1"

    t2 = policy.next_turns("sess_rr", dummy_snap, agents)[0]
    assert t2.agent_id == "agent_2"

    t3 = policy.next_turns("sess_rr", dummy_snap, agents)[0]
    assert t3.agent_id == "agent_3"

    t4 = policy.next_turns("sess_rr", dummy_snap, agents)[0]
    assert t4.agent_id == "agent_1"


def test_reactive_pxp_policy_flow():
    policy = ReactivePXPPolicy()
    agents = {
        "agent_primary": {"role": AgentRole.PRIMARY, "priority": 1},
        "agent_critic": {"role": AgentRole.CRITIC, "priority": 1},
        "agent_expert": {"role": AgentRole.DOMAIN_EXPERT, "priority": 2},
    }
    snap = BlackboardSnapshot(
        session_id="sess_reactive",
        task_id="t_pxp",
        task_description="Clinical task",
    )

    # 1. Empty board: schedules Primary
    turns = policy.next_turns("sess_reactive", snap, agents, last_contribution=None)
    assert len(turns) == 1
    assert turns[0].agent_id == "agent_primary"

    # 2. Primary PROPOSE: schedules Critic (priority 1) and Expert (priority 2)
    contrib_propose = AgentContribution(
        contribution_id="c_prop",
        session_id="sess_reactive",
        turn_index=1,
        agent_id="agent_primary",
        tag=PXPTag.REVISE,  # PROPOSE maps to REVISE
        payload=PEXPayload(
            prediction=Prediction(claim="Diagnosis A", confidence=0.8),
            explanation=Explanation(rationale="Initial hypothesis"),
        ),
    )
    snap.contributions.append(contrib_propose)

    turns = policy.next_turns("sess_reactive", snap, agents, last_contribution=contrib_propose)
    agent_ids = [t.agent_id for t in turns]
    assert "agent_critic" in agent_ids
    assert "agent_expert" in agent_ids

    # 3. Critic REFUTE: schedules Primary back to defend
    contrib_refute = AgentContribution(
        contribution_id="c_refute",
        session_id="sess_reactive",
        turn_index=2,
        agent_id="agent_critic",
        target_contribution_id="c_prop",
        tag=PXPTag.REFUTE,
        payload=PEXPayload(
            prediction=Prediction(claim="Diagnosis B", confidence=0.7),
            explanation=Explanation(rationale="Disputing diagnosis A"),
        ),
    )
    snap.contributions.append(contrib_refute)

    turns = policy.next_turns("sess_reactive", snap, agents, last_contribution=contrib_refute)
    assert len(turns) == 1
    assert turns[0].agent_id == "agent_primary"


# ==============================================================================
# 3. DeterministicScheduler Integration with Redis Storage
# ==============================================================================

@pytest.fixture
def redis_store():
    mock_client = InMemoryRedisClient()
    return RedisBlackboardStore(redis_client=mock_client)


def test_scheduler_step_with_mock_agents_and_redis_locks(redis_store: RedisBlackboardStore):
    session_id = "sess_sched_001"
    redis_store.create_session(session_id, "task_sched_1", "Debate prompt")

    scheduler = DeterministicScheduler(
        session_id=session_id,
        store=redis_store,
        policy=RoundRobinPolicy(),
    )

    # Register two mock agents
    broker1 = MockLLMBroker(default_response={
        "tag": "RATIFY",
        "prediction": "Diagnosis A",
        "explanation": "Rationale from Agent 1",
        "confidence": 0.85,
    })
    agent1 = MockPEXAgent(agent_id="agent_1", agent_role="Primary", broker=broker1)

    broker2 = MockLLMBroker(default_response={
        "tag": "RATIFY",
        "prediction": "Diagnosis A",
        "explanation": "Rationale from Agent 2",
        "confidence": 0.90,
    })
    agent2 = MockPEXAgent(agent_id="agent_2", agent_role="Critic", broker=broker2)

    scheduler.register_agent("agent_1", AgentRole.PRIMARY, agent1)
    scheduler.register_agent("agent_2", AgentRole.CRITIC, agent2)

    # Execute step 1 (Agent 1)
    c1 = scheduler.step()
    assert c1 is not None
    assert c1.agent_id == "agent_1"
    assert c1.turn_index == 1

    snap1 = redis_store.get_snapshot(session_id)
    assert snap1.version == 1
    assert len(snap1.contributions) == 1

    # Execute step 2 (Agent 2)
    c2 = scheduler.step()
    assert c2 is not None
    assert c2.agent_id == "agent_2"
    assert c2.turn_index == 2

    snap2 = redis_store.get_snapshot(session_id)
    assert snap2.version == 2
    assert len(snap2.contributions) == 2


def test_scheduler_consensus_termination(redis_store: RedisBlackboardStore):
    session_id = "sess_consensus_001"
    redis_store.create_session(session_id, "task_con", "Consensus task")

    scheduler = DeterministicScheduler(
        session_id=session_id,
        store=redis_store,
        policy=RoundRobinPolicy(),
        consensus_threshold=2,
    )

    # Both agents return RATIFY
    broker = MockLLMBroker(default_response={
        "tag": "RATIFY",
        "prediction": "Consensus Claim",
        "explanation": "Full clinical agreement.",
        "confidence": 0.95,
    })
    agent_a = MockPEXAgent(agent_id="agent_a", agent_role="primary", broker=broker)
    agent_b = MockPEXAgent(agent_id="agent_b", agent_role="critic", broker=broker)

    scheduler.register_agent("agent_a", AgentRole.PRIMARY, agent_a)
    scheduler.register_agent("agent_b", AgentRole.CRITIC, agent_b)

    final_snapshot = scheduler.run(max_turns=5)
    assert final_snapshot.status == BlackboardStatus.CONSENSUS
    assert len(final_snapshot.contributions) == 2
    assert final_snapshot.active_claim is not None
    assert final_snapshot.active_claim.claim == "Consensus Claim"


def test_scheduler_deadlock_detection_and_hook(redis_store: RedisBlackboardStore):
    session_id = "sess_deadlock_001"
    redis_store.create_session(session_id, "task_dl", "Deadlock task")

    scheduler = DeterministicScheduler(
        session_id=session_id,
        store=redis_store,
        policy=RoundRobinPolicy(),
        deadlock_window=2,
    )

    # Broker produces disagreement: Agent 1 refutes, Agent 2 rejects
    broker1 = MockLLMBroker(default_response={
        "tag": "REFUTE",
        "prediction": "Claim Alpha",
        "explanation": "Refuting previous claim",
        "confidence": 0.75,
    })
    broker2 = MockLLMBroker(default_response={
        "tag": "REJECT",
        "prediction": "Claim Beta",
        "explanation": "Fundamentally rejecting",
        "confidence": 0.80,
    })
    agent1 = MockPEXAgent(agent_id="agent_alpha", agent_role="primary", broker=broker1)
    agent2 = MockPEXAgent(agent_id="agent_beta", agent_role="critic", broker=broker2)

    scheduler.register_agent("agent_alpha", AgentRole.PRIMARY, agent1)
    scheduler.register_agent("agent_beta", AgentRole.CRITIC, agent2)

    received_deadlocks = []

    def deadlock_hook(notif: DeadlockNotification):
        received_deadlocks.append(notif)
        return None  # No immediate resolution, let scheduler pause

    scheduler.register_deadlock_hook(deadlock_hook)

    # Step 1: Alpha refutes
    c1 = scheduler.step()
    assert c1.tag == PXPTag.REFUTE
    assert scheduler.queue.is_paused is False

    # Step 2: Beta rejects -> Deadlock window of 2 across 2 agents triggers!
    c2 = scheduler.step()
    assert c2.tag == PXPTag.REJECT
    assert scheduler.queue.is_paused is True

    snap = redis_store.get_snapshot(session_id)
    assert snap.status == BlackboardStatus.DEADLOCK
    assert len(received_deadlocks) == 1
    assert "agent_alpha" in received_deadlocks[0].conflicting_agent_ids
    assert "agent_beta" in received_deadlocks[0].conflicting_agent_ids


def test_deadlock_hook_with_counterfactual_recovery(redis_store: RedisBlackboardStore):
    session_id = "sess_recovery_001"
    redis_store.create_session(session_id, "task_rec", "Recovery task")

    scheduler = DeterministicScheduler(
        session_id=session_id,
        store=redis_store,
        policy=RoundRobinPolicy(),
        deadlock_window=2,
    )

    broker1 = MockLLMBroker(default_response={"tag": "REFUTE", "prediction": "P1", "explanation": "E1"})
    broker2 = MockLLMBroker(default_response={"tag": "REJECT", "prediction": "P2", "explanation": "E2"})

    agent1 = MockPEXAgent(agent_id="agent_1", agent_role="primary", broker=broker1)
    agent2 = MockPEXAgent(agent_id="agent_2", agent_role="critic", broker=broker2)

    scheduler.register_agent("agent_1", AgentRole.PRIMARY, agent1)
    scheduler.register_agent("agent_2", AgentRole.CRITIC, agent2)

    # Hook that simulates Student 3's counterfactual sandbox resolving the deadlock
    def counterfactual_resolver(notif: DeadlockNotification) -> ReviseProposal:
        return ReviseProposal(
            session_id=notif.session_id,
            winning_branch_id="cf_sandbox_branch_1",
            target_turn_index=2,
            revised_contribution=AgentContribution(
                contribution_id="c_recovery_turn",
                session_id=notif.session_id,
                turn_index=3,
                agent_id="counterfactual_agent",
                agent_role=AgentRole.COUNTERFACTUAL,
                tag=PXPTag.REVISE,
                payload=PEXPayload(
                    prediction=Prediction(claim="Synthesized Unified Claim", confidence=0.96),
                    explanation=Explanation(rationale="Resolved conflicting perspectives via sandbox replay."),
                ),
            ),
            credit_assignment_summary="Backtracked to divergence node and synthesized compromise.",
        )

    scheduler.register_deadlock_hook(counterfactual_resolver)

    # Step 1: Agent 1 refutes
    scheduler.step()
    # Step 2: Agent 2 rejects -> Deadlock triggered -> Hook returns ReviseProposal -> Injected at Priority 0!
    scheduler.step()

    # Scheduler unpaused and recovery turn queued at Priority 0
    assert scheduler.queue.is_paused is False
    assert len(scheduler.queue) >= 1
    assert scheduler.queue.peek().priority == 0
    assert scheduler.queue.peek().agent_id == "counterfactual_agent"

    # Step 3: Executes the injected counterfactual proposal
    c3 = scheduler.step()
    assert c3 is not None
    assert c3.agent_id == "counterfactual_agent"
    assert c3.tag == PXPTag.REVISE
    assert c3.is_counterfactual is True
    assert c3.prediction == "Synthesized Unified Claim"

    snap = redis_store.get_snapshot(session_id)
    assert snap.status == BlackboardStatus.ACTIVE
    assert len(snap.contributions) == 3


def test_scheduler_max_turns_guard(redis_store: RedisBlackboardStore):
    session_id = "sess_max_turns_001"
    redis_store.create_session(session_id, "task_mt", "Infinite debate guard")

    # Round robin between two agents without deadlock (single agent refuting alternating)
    scheduler = DeterministicScheduler(
        session_id=session_id,
        store=redis_store,
        policy=RoundRobinPolicy(),
        deadlock_window=5,  # high window to avoid deadlock
    )

    broker = MockLLMBroker(default_response={"tag": "REVISE", "prediction": "P", "explanation": "E"})
    agent = MockPEXAgent(agent_id="agent_solo", broker=broker)
    scheduler.register_agent("agent_solo", AgentRole.PRIMARY, agent)

    final_snap = scheduler.run(max_turns=4)
    assert len(final_snap.contributions) == 4
