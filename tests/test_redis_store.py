"""
tests/test_redis_store.py
Comprehensive unit, concurrency, and integration tests for RedisBlackboardStore.
Validates state persistence, distributed write-locking (SET NX PX + safe Lua release),
optimistic concurrency control (OCC), multi-threaded concurrent write safety,
deadlock detection heuristics, sandbox cloning, and telemetry publishing.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import List
import pytest

from contracts.schemas import (
    AgentContribution,
    AgentRole,
    BlackboardSnapshot,
    BlackboardStatus,
    Explanation,
    LockAcquireRequest,
    LockReleaseRequest,
    PEXPayload,
    Prediction,
    PXPTag,
    StateWriteRequest,
    TelemetryEventType,
)
from storage.exceptions import (
    LockAcquisitionError,
    LockNotHeldError,
    SessionNotFoundError,
    VersionConflictError,
)
from storage.redis_store import InMemoryRedisClient, RedisBlackboardStore


@pytest.fixture
def store():
    """Provides a clean in-memory RedisBlackboardStore instance."""
    mock_client = InMemoryRedisClient()
    return RedisBlackboardStore(redis_client=mock_client)


def _make_contribution(
    turn_index: int,
    agent_id: str,
    tag: PXPTag = PXPTag.RATIFY,
    claim: str = "Test Diagnosis",
    branch_id: str = "main",
) -> AgentContribution:
    return AgentContribution(
        contribution_id=f"c_{agent_id}_{turn_index}_{int(time.time()*1000)}",
        session_id="test_session",
        turn_index=turn_index,
        agent_id=agent_id,
        tag=tag,
        payload=PEXPayload(
            prediction=Prediction(claim=claim, confidence=0.88),
            explanation=Explanation(rationale=f"Rationale by {agent_id}", evidence=[]),
        ),
        metadata={"branch_id": branch_id},
    )


# ==============================================================================
# 1. Session Lifecycle Tests
# ==============================================================================

def test_session_lifecycle(store: RedisBlackboardStore):
    session_id = "sess_med_001"
    snapshot = store.create_session(
        session_id=session_id,
        task_id="task_med_1",
        task_description="Patient presents with shortness of breath.",
        ground_truth="IPF",
        initial_context={"vital_signs": "BP 120/80"},
    )

    assert snapshot.session_id == session_id
    assert snapshot.task_id == "task_med_1"
    assert snapshot.task_description == "Patient presents with shortness of breath."
    assert snapshot.version == 0
    assert snapshot.status == BlackboardStatus.ACTIVE
    assert snapshot.metadata.get("ground_truth") == "IPF"
    assert snapshot.initial_context == {"vital_signs": "BP 120/80"}
    assert store.session_exists(session_id) is True

    # Retrieve snapshot and verify fidelity
    loaded = store.get_snapshot(session_id)
    assert loaded.session_id == session_id
    assert loaded.version == 0
    assert loaded.contributions == []

    # Non-existent session
    with pytest.raises(SessionNotFoundError):
        store.get_snapshot("non_existent_session")

    # Delete session
    assert store.delete_session(session_id) is True
    assert store.session_exists(session_id) is False


# ==============================================================================
# 2. Distributed Lock Mutual Exclusion & TTL
# ==============================================================================

def test_lock_mutual_exclusion(store: RedisBlackboardStore):
    session_id = "sess_lock_001"
    store.create_session(session_id, "t1", "Problem 1")

    # Agent 1 acquires lock
    req1 = LockAcquireRequest(session_id=session_id, agent_id="agent_1", timeout_seconds=5.0)
    resp1 = store.acquire_lock(req1)
    assert resp1.acquired is True
    assert resp1.lock_token is not None
    assert resp1.holder_id == "agent_1"

    # Agent 2 attempts to acquire lock while held by Agent 1
    req2 = LockAcquireRequest(session_id=session_id, agent_id="agent_2", timeout_seconds=5.0)
    resp2 = store.acquire_lock(req2)
    assert resp2.acquired is False
    assert resp2.lock_token is None
    assert resp2.holder_id == "agent_1"
    assert "currently held by agent 'agent_1'" in (resp2.error_message or "")


def test_lock_release_and_reacquisition(store: RedisBlackboardStore):
    session_id = "sess_lock_002"
    store.create_session(session_id, "t2", "Problem 2")

    # Agent 1 acquires and releases
    req1 = LockAcquireRequest(session_id=session_id, agent_id="agent_1", timeout_seconds=5.0)
    resp1 = store.acquire_lock(req1)
    assert resp1.acquired is True

    rel_req = LockReleaseRequest(
        session_id=session_id,
        agent_id="agent_1",
        lock_token=resp1.lock_token,
    )
    assert store.release_lock(rel_req) is True

    # Agent 2 can now acquire lock
    req2 = LockAcquireRequest(session_id=session_id, agent_id="agent_2", timeout_seconds=5.0)
    resp2 = store.acquire_lock(req2)
    assert resp2.acquired is True
    assert resp2.holder_id == "agent_2"


def test_safe_lua_release_prevents_stale_release(store: RedisBlackboardStore):
    session_id = "sess_lock_003"
    store.create_session(session_id, "t3", "Problem 3")

    # Agent 1 acquires lock
    resp1 = store.acquire_lock(LockAcquireRequest(session_id=session_id, agent_id="agent_1", timeout_seconds=5.0))
    assert resp1.acquired is True

    # Agent 2 attempts to release Agent 1's lock with mismatched token
    rel_req_fake = LockReleaseRequest(
        session_id=session_id,
        agent_id="agent_2",
        lock_token="fake_token_123",
    )
    assert store.release_lock(rel_req_fake) is False

    # Lock is still held by Agent 1
    resp2 = store.acquire_lock(LockAcquireRequest(session_id=session_id, agent_id="agent_2", timeout_seconds=5.0))
    assert resp2.acquired is False
    assert resp2.holder_id == "agent_1"


def test_lock_lease_ttl_expiration(store: RedisBlackboardStore):
    session_id = "sess_lock_004"
    store.create_session(session_id, "t4", "Problem 4")

    # Agent 1 acquires lock with ultra-short TTL (100ms)
    resp1 = store.acquire_lock(LockAcquireRequest(session_id=session_id, agent_id="agent_1", timeout_seconds=0.1))
    assert resp1.acquired is True

    # Sleep past expiration
    time.sleep(0.15)

    # Agent 2 acquires lock after Agent 1's lease expired
    resp2 = store.acquire_lock(LockAcquireRequest(session_id=session_id, agent_id="agent_2", timeout_seconds=5.0))
    assert resp2.acquired is True
    assert resp2.holder_id == "agent_2"


# ==============================================================================
# 3. Transactional Write & Optimistic Concurrency Control
# ==============================================================================

def test_write_state_success(store: RedisBlackboardStore):
    session_id = "sess_write_001"
    store.create_session(session_id, "t5", "Problem 5")

    # Acquire lock
    resp = store.acquire_lock(LockAcquireRequest(session_id=session_id, agent_id="agent_1", timeout_seconds=5.0))
    assert resp.acquired is True

    # Submit turn at expected_version=0
    contrib = _make_contribution(turn_index=1, agent_id="agent_1", tag=PXPTag.RATIFY, claim="Consensus Diagnosed")
    write_req = StateWriteRequest(
        session_id=session_id,
        agent_id="agent_1",
        lock_token=resp.lock_token,
        expected_version=0,
        contribution=contrib,
    )

    updated_snapshot = store.write_state(write_req)
    assert updated_snapshot.version == 1
    assert len(updated_snapshot.contributions) == 1
    assert updated_snapshot.contributions[0].agent_id == "agent_1"
    assert updated_snapshot.active_claim is not None
    assert updated_snapshot.active_claim.claim == "Consensus Diagnosed"
    assert "agent_1" in updated_snapshot.active_agents

    # Release lock
    store.release_lock(LockReleaseRequest(session_id=session_id, agent_id="agent_1", lock_token=resp.lock_token))


def test_write_state_fails_without_valid_lock(store: RedisBlackboardStore):
    session_id = "sess_write_002"
    store.create_session(session_id, "t6", "Problem 6")

    contrib = _make_contribution(turn_index=1, agent_id="agent_1")
    write_req = StateWriteRequest(
        session_id=session_id,
        agent_id="agent_1",
        lock_token="unacquired_token",
        expected_version=0,
        contribution=contrib,
    )

    with pytest.raises(LockNotHeldError):
        store.write_state(write_req)


def test_write_state_version_conflict_occ(store: RedisBlackboardStore):
    session_id = "sess_write_003"
    store.create_session(session_id, "t7", "Problem 7")

    # Step 1: Write version 1
    with store.lock_context(session_id, "agent_1") as token1:
        contrib1 = _make_contribution(turn_index=1, agent_id="agent_1")
        store.write_state(StateWriteRequest(
            session_id=session_id,
            agent_id="agent_1",
            lock_token=token1,
            expected_version=0,
            contribution=contrib1,
        ))

    # Step 2: Attempt to write with stale expected_version=0
    with store.lock_context(session_id, "agent_2") as token2:
        contrib2 = _make_contribution(turn_index=2, agent_id="agent_2")
        stale_request = StateWriteRequest(
            session_id=session_id,
            agent_id="agent_2",
            lock_token=token2,
            expected_version=0,  # Stale! Current is 1
            contribution=contrib2,
        )
        with pytest.raises(VersionConflictError) as exc_info:
            store.write_state(stale_request)

        assert exc_info.value.expected_version == 0
        assert exc_info.value.current_version == 1


# ==============================================================================
# 4. Multi-Threaded Concurrent Writer Stress Test
# ==============================================================================

def test_concurrent_writers_no_lost_updates(store: RedisBlackboardStore):
    """
    Simulates 8 concurrent agent threads competing to submit turns.
    Verifies that:
    1. Locks prevent collision / overwriting.
    2. Version counter increments strictly and monotonically without gaps.
    3. Exactly 8 contributions are persisted in history.
    """
    session_id = "sess_concurrent_001"
    store.create_session(session_id, "t_conc", "Multi-Agent Concurrent Test")

    num_agents = 8
    success_counts = []

    def agent_worker(agent_idx: int) -> int:
        agent_id = f"agent_{agent_idx}"
        max_attempts = 30
        for _ in range(max_attempts):
            try:
                # 1. Read current snapshot to get expected_version
                current = store.get_snapshot(session_id)
                expected_v = current.version

                # 2. Acquire lock with context manager (retries internally)
                with store.lock_context(session_id, agent_id, timeout_seconds=2.0, max_retries=15):
                    # Re-verify version under lock
                    latest = store.get_snapshot(session_id)
                    expected_v = latest.version

                    contrib = _make_contribution(
                        turn_index=expected_v + 1,
                        agent_id=agent_id,
                        claim=f"Claim from {agent_id} at step {expected_v + 1}",
                    )
                    store.write_state(StateWriteRequest(
                        session_id=session_id,
                        agent_id=agent_id,
                        lock_token=store._client.get(store._lock_key(session_id)).split(":")[1],
                        expected_version=expected_v,
                        contribution=contrib,
                    ))
                    return 1
            except (LockAcquisitionError, VersionConflictError):
                time.sleep(0.01)
                continue
        return 0

    with ThreadPoolExecutor(max_workers=num_agents) as executor:
        futures = [executor.submit(agent_worker, i) for i in range(num_agents)]
        for f in as_completed(futures):
            success_counts.append(f.result())

    final_snapshot = store.get_snapshot(session_id)
    assert sum(success_counts) == num_agents
    assert final_snapshot.version == num_agents
    assert len(final_snapshot.contributions) == num_agents
    # Verify no duplicate turn indices
    turn_indices = [c.turn_index for c in final_snapshot.contributions]
    assert len(set(turn_indices)) == num_agents


# ==============================================================================
# 5. Deadlock Detection & Sandbox History Cloning
# ==============================================================================

def test_deadlock_detection(store: RedisBlackboardStore):
    session_id = "sess_deadlock_001"
    store.create_session(session_id, "t_dl", "Deadlock detection scenario")

    # Add 2 REFUTE and 1 REJECT turns
    tags = [PXPTag.REFUTE, PXPTag.REFUTE, PXPTag.REJECT]
    for idx, tag in enumerate(tags, start=1):
        with store.lock_context(session_id, f"agent_{idx}") as token:
            snap = store.get_snapshot(session_id)
            c = _make_contribution(turn_index=idx, agent_id=f"agent_{idx}", tag=tag)
            store.write_state(StateWriteRequest(
                session_id=session_id,
                agent_id=f"agent_{idx}",
                lock_token=token,
                expected_version=snap.version,
                contribution=c,
            ))

    # Deadlock window 3 should trigger
    assert store.detect_deadlock(session_id, window=3) is True
    # Window 4 requires more entries
    assert store.detect_deadlock(session_id, window=4) is False


def test_sandbox_history_cloning(store: RedisBlackboardStore):
    session_id = "sess_main_001"
    store.create_session(session_id, "t_sbx", "Main debate with divergence")

    # Write 4 turns on main branch
    for idx in range(1, 5):
        with store.lock_context(session_id, f"agent_{idx}") as token:
            snap = store.get_snapshot(session_id)
            c = _make_contribution(turn_index=idx, agent_id=f"agent_{idx}")
            store.write_state(StateWriteRequest(
                session_id=session_id,
                agent_id=f"agent_{idx}",
                lock_token=token,
                expected_version=snap.version,
                contribution=c,
            ))

    # Clone up to step 2 for Student 3's counterfactual sandbox
    sandbox_session_id = "sess_sandbox_001"
    sandbox_branch = "cf_branch_alpha"
    cloned_snap = store.clone_sandbox(
        source_session_id=session_id,
        target_session_id=sandbox_session_id,
        sandbox_branch_id=sandbox_branch,
        up_to_step=2,
    )

    assert cloned_snap.session_id == sandbox_session_id
    assert cloned_snap.version == 2
    assert len(cloned_snap.contributions) == 2
    assert cloned_snap.contributions[0].branch_id == sandbox_branch
    assert cloned_snap.contributions[1].branch_id == sandbox_branch
    assert cloned_snap.metadata.get("cloned_from") == session_id

    # Original session remains unaffected
    orig = store.get_snapshot(session_id)
    assert orig.version == 4
    assert len(orig.contributions) == 4


# ==============================================================================
# 6. Pub/Sub Telemetry Streaming
# ==============================================================================

def test_telemetry_pubsub_broadcasting(store: RedisBlackboardStore):
    session_id = "sess_telemetry_001"
    sub_queue = store.client.subscribe(f"blackboard:{session_id}:events")

    # Create session emits BOARD_INITIALIZED
    store.create_session(session_id, "t_pub", "Pub/sub test")
    assert len(sub_queue) >= 1
    init_event = sub_queue[0]
    assert "BOARD_INITIALIZED" in init_event["data"]

    # Write state emits AGENT_SUBMISSION
    with store.lock_context(session_id, "agent_tele") as token:
        c = _make_contribution(turn_index=1, agent_id="agent_tele")
        store.write_state(StateWriteRequest(
            session_id=session_id,
            agent_id="agent_tele",
            lock_token=token,
            expected_version=0,
            contribution=c,
        ))

    # Check all events published in the sequence:
    # 1. BOARD_INITIALIZED, 2. LOCK_ACQUIRED, 3. AGENT_SUBMISSION, 4. LOCK_RELEASED
    assert len(sub_queue) >= 4
    event_payloads = [e["data"] for e in sub_queue]
    assert any("BOARD_INITIALIZED" in data for data in event_payloads)
    assert any("LOCK_ACQUIRED" in data for data in event_payloads)
    assert any("AGENT_SUBMISSION" in data for data in event_payloads)
    assert any("LOCK_RELEASED" in data for data in event_payloads)
