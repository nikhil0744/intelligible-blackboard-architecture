"""
storage/redis_store.py
Production-grade Redis-backed transactional Blackboard repository and distributed lock manager.
Provides dual-layer concurrency protection: exclusive distributed locks (SET NX PX + Lua)
combined with optimistic concurrency control (OCC version checks) to prevent agent write races.
Includes an in-memory fallback client for offline/mock test execution.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, Generator, List, Optional
from uuid import uuid4

import redis

from contracts.schemas import (
    AgentContribution,
    AgentRole,
    BlackboardEntry,
    BlackboardSnapshot,
    BlackboardStatus,
    Explanation,
    IntelligibilityLevel,
    LockAcquireRequest,
    LockAcquireResponse,
    LockReleaseRequest,
    PEXPayload,
    Prediction,
    PXPTag,
    StateWriteRequest,
    TelemetryEvent,
    TelemetryEventType,
)
from storage.exceptions import (
    BlackboardStorageError,
    LockAcquisitionError,
    LockNotHeldError,
    SessionNotFoundError,
    VersionConflictError,
)

logger = logging.getLogger(__name__)

# Safe Lua script for releasing locks only if the current token matches the holder's token
LUA_RELEASE_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""


class InMemoryRedisClient:
    """
    Thread-safe, in-memory mock Redis client that emulates key-value storage,
    atomic SET NX PX, Lua script evaluation, and Pub/Sub mechanics for offline
    development and deterministic unit testing.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._data: Dict[str, str] = {}
        self._expirations: Dict[str, float] = {}
        self._subscribers: Dict[str, List[Any]] = {}

    def _purge_if_expired(self, key: str) -> None:
        if key in self._expirations:
            if time.time() >= self._expirations[key]:
                self._data.pop(key, None)
                self._expirations.pop(key, None)

    def ping(self) -> bool:
        return True

    def get(self, name: str) -> Optional[str]:
        with self._lock:
            self._purge_if_expired(name)
            return self._data.get(name)

    def set(
        self,
        name: str,
        value: Any,
        ex: Optional[int | float] = None,
        px: Optional[int | float] = None,
        nx: bool = False,
        xx: bool = False,
    ) -> bool:
        with self._lock:
            self._purge_if_expired(name)
            exists = name in self._data

            if nx and exists:
                return False
            if xx and not exists:
                return False

            self._data[name] = str(value)
            if px is not None:
                self._expirations[name] = time.time() + (px / 1000.0)
            elif ex is not None:
                self._expirations[name] = time.time() + float(ex)
            else:
                self._expirations.pop(name, None)
            return True

    def delete(self, *names: str) -> int:
        with self._lock:
            count = 0
            for name in names:
                self._purge_if_expired(name)
                if name in self._data:
                    del self._data[name]
                    self._expirations.pop(name, None)
                    count += 1
            return count

    def exists(self, *names: str) -> int:
        with self._lock:
            count = 0
            for name in names:
                self._purge_if_expired(name)
                if name in self._data:
                    count += 1
            return count

    def eval(self, script: str, numkeys: int, *keys_and_args: Any) -> Any:
        with self._lock:
            keys = keys_and_args[:numkeys]
            args = keys_and_args[numkeys:]

            # Emulate standard LUA release script
            if "if redis.call('get', KEYS[1]) == ARGV[1]" in script:
                key = keys[0]
                expected_val = str(args[0])
                current_val = self.get(key)
                if current_val == expected_val:
                    self.delete(key)
                    return 1
                return 0

            raise NotImplementedError("Arbitrary Lua script execution not implemented in InMemoryRedisClient")

    def publish(self, channel: str, message: str) -> int:
        with self._lock:
            subs = self._subscribers.get(channel, [])
            for queue in subs:
                queue.append({"channel": channel, "data": message})
            return len(subs)

    def subscribe(self, channel: str) -> List[Dict[str, str]]:
        with self._lock:
            if channel not in self._subscribers:
                self._subscribers[channel] = []
            queue: List[Dict[str, str]] = []
            self._subscribers[channel].append(queue)
            return queue

    def flushall(self) -> bool:
        with self._lock:
            self._data.clear()
            self._expirations.clear()
            self._subscribers.clear()
            return True


class RedisBlackboardStore:
    """
    Transactional Redis storage engine for multi-agent blackboard sessions.
    
    Responsibilities:
    1. Session persistence and serialization of BlackboardSnapshot objects.
    2. Distributed write locking using Redis SET NX PX and Lua-verified atomic releases.
    3. Optimistic Concurrency Control (OCC) validating version counters.
    4. Real-time telemetry event streaming over Redis Pub/Sub channels.
    5. Retrospective history cloning for counterfactual sandboxes.
    """

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        redis_url: Optional[str] = None,
        auto_fallback: bool = True,
    ):
        """
        Initialize the Redis store.
        If redis_client is provided, it is used directly.
        Otherwise, attempts connection to redis_url (or REDIS_URL env variable).
        If connection fails and auto_fallback is True, falls back to InMemoryRedisClient.
        """
        self._url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        self._client: Any = None
        self._is_in_memory: bool = False

        if redis_client is not None:
            self._client = redis_client
            self._is_in_memory = isinstance(redis_client, InMemoryRedisClient)
        elif os.environ.get("REDIS_MODE", "").lower() == "memory":
            self._client = InMemoryRedisClient()
            self._is_in_memory = True
        else:
            try:
                client = redis.from_url(self._url, decode_responses=True)
                client.ping()
                self._client = client
                self._is_in_memory = False
            except Exception as e:
                if auto_fallback:
                    logger.warning(
                        "Failed to connect to Redis at %s (%s). Falling back to InMemoryRedisClient.",
                        self._url,
                        e,
                    )
                    self._client = InMemoryRedisClient()
                    self._is_in_memory = True
                else:
                    raise

    @property
    def client(self) -> Any:
        return self._client

    @property
    def is_in_memory(self) -> bool:
        return self._is_in_memory

    # --------------------------------------------------------------------------
    # Redis Key Helpers
    # --------------------------------------------------------------------------

    def _state_key(self, session_id: str) -> str:
        return f"blackboard:{session_id}:state"

    def _lock_key(self, session_id: str) -> str:
        return f"blackboard:{session_id}:lock"

    def _version_key(self, session_id: str) -> str:
        return f"blackboard:{session_id}:version"

    def _events_channel(self, session_id: str) -> str:
        return f"blackboard:{session_id}:events"

    # --------------------------------------------------------------------------
    # Session Lifecycle Management
    # --------------------------------------------------------------------------

    def create_session(
        self,
        session_id: str,
        task_id: str,
        task_description: str,
        ground_truth: Optional[str] = None,
        initial_context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BlackboardSnapshot:
        """
        Creates and persists a new blackboard reasoning session.
        Overwrites any existing session with the same session_id.
        """
        meta = metadata.copy() if metadata else {}
        if ground_truth is not None:
            meta["ground_truth"] = ground_truth
        if "active_branches" not in meta:
            meta["active_branches"] = ["main"]

        snapshot = BlackboardSnapshot(
            session_id=session_id,
            task_id=task_id,
            task_description=task_description,
            initial_context=initial_context or {},
            status=BlackboardStatus.ACTIVE,
            version=0,
            active_claim=None,
            contributions=[],
            active_agents=[],
            metadata=meta,
        )

        self._save_snapshot(snapshot)
        self._client.set(self._version_key(session_id), "0")

        # Emit initialization telemetry
        self._emit_telemetry(
            session_id=session_id,
            event_type=TelemetryEventType.BOARD_INITIALIZED,
            payload={
                "task_id": task_id,
                "problem_statement": task_description,
                "active_branches": snapshot.active_branches,
            },
        )
        return snapshot

    def session_exists(self, session_id: str) -> bool:
        """Checks if a session exists in the store."""
        return bool(self._client.exists(self._state_key(session_id)))

    def get_snapshot(self, session_id: str) -> BlackboardSnapshot:
        """Retrieves the full blackboard snapshot for a session."""
        raw_state = self._client.get(self._state_key(session_id))
        if not raw_state:
            raise SessionNotFoundError(session_id)
        return BlackboardSnapshot.model_validate_json(raw_state)

    def delete_session(self, session_id: str) -> bool:
        """Deletes all Redis keys associated with a session."""
        keys = [
            self._state_key(session_id),
            self._lock_key(session_id),
            self._version_key(session_id),
        ]
        return bool(self._client.delete(*keys))

    def _save_snapshot(self, snapshot: BlackboardSnapshot) -> None:
        """Serializes and writes the snapshot to Redis."""
        self._client.set(self._state_key(snapshot.session_id), snapshot.model_dump_json())

    # --------------------------------------------------------------------------
    # Distributed Write Lock Management
    # --------------------------------------------------------------------------

    def acquire_lock(self, request: LockAcquireRequest) -> LockAcquireResponse:
        """
        Attempts to acquire an exclusive write lease using Redis SET NX PX.
        Returns LockAcquireResponse indicating success or failure.
        """
        lock_key = self._lock_key(request.session_id)
        token = uuid4().hex
        lock_val = f"{request.agent_id}:{token}"
        px_ms = int(request.timeout_seconds * 1000)

        # Atomic SET with NX (not exists) and PX (millisecond TTL)
        acquired = bool(self._client.set(lock_key, lock_val, px=px_ms, nx=True))

        if acquired:
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=request.timeout_seconds)
            self._emit_telemetry(
                session_id=request.session_id,
                event_type=TelemetryEventType.LOCK_ACQUIRED,
                agent_id=request.agent_id,
                payload={"lock_token": token, "timeout_seconds": request.timeout_seconds},
            )
            return LockAcquireResponse(
                acquired=True,
                lock_token=token,
                holder_id=request.agent_id,
                expires_at=expires_at,
                error_message=None,
            )

        # Lock acquisition failed: Inspect existing holder
        current_val = self._client.get(lock_key)
        holder_id = current_val.split(":")[0] if current_val and ":" in current_val else None

        return LockAcquireResponse(
            acquired=False,
            lock_token=None,
            holder_id=holder_id,
            expires_at=None,
            error_message=f"Lock currently held by agent '{holder_id}'.",
        )

    def release_lock(self, request: LockReleaseRequest) -> bool:
        """
        Releases an exclusive write lock using an atomic Lua script to ensure
        an agent only deletes the key if its token matches the active lease.
        """
        lock_key = self._lock_key(request.session_id)
        lock_val = f"{request.agent_id}:{request.lock_token}"

        result = self._client.eval(LUA_RELEASE_SCRIPT, 1, lock_key, lock_val)
        released = bool(result == 1)

        if released:
            self._emit_telemetry(
                session_id=request.session_id,
                event_type=TelemetryEventType.LOCK_RELEASED,
                agent_id=request.agent_id,
                payload={"lock_token": request.lock_token},
            )
        return released

    @contextmanager
    def lock_context(
        self,
        session_id: str,
        agent_id: str,
        timeout_seconds: float = 5.0,
        retry_interval_ms: int = 50,
        max_retries: int = 10,
    ) -> Generator[str, None, None]:
        """
        Pythonic context manager for acquiring and safely releasing a write lock.
        Raises LockAcquisitionError if lock cannot be acquired within retry bounds.
        """
        request = LockAcquireRequest(
            session_id=session_id,
            agent_id=agent_id,
            timeout_seconds=timeout_seconds,
        )

        response = self.acquire_lock(request)
        attempts = 0
        while not response.acquired and attempts < max_retries:
            time.sleep(retry_interval_ms / 1000.0)
            attempts += 1
            response = self.acquire_lock(request)

        if not response.acquired or not response.lock_token:
            raise LockAcquisitionError(
                session_id=session_id,
                agent_id=agent_id,
                holder_id=response.holder_id,
            )

        token = response.lock_token
        try:
            yield token
        finally:
            self.release_lock(
                LockReleaseRequest(
                    session_id=session_id,
                    agent_id=agent_id,
                    lock_token=token,
                )
            )

    # --------------------------------------------------------------------------
    # Transactional State Mutation (OCC + Lock Verified)
    # --------------------------------------------------------------------------

    def write_state(self, request: StateWriteRequest) -> BlackboardSnapshot:
        """
        Appends an agent contribution to the blackboard under strict concurrency controls:
        1. Validates that the agent holds the active lock matching lock_token.
        2. Validates that expected_version matches the current snapshot version.
        3. Appends contribution, increments version, updates active claim, and persists state.
        4. Broadcasts telemetry event to Redis Pub/Sub.
        """
        session_id = request.session_id
        lock_key = self._lock_key(session_id)
        current_lock_val = self._client.get(lock_key)

        expected_lock_val = f"{request.agent_id}:{request.lock_token}"
        if current_lock_val != expected_lock_val:
            raise LockNotHeldError(
                session_id=session_id,
                agent_id=request.agent_id,
                detail=f"Lock token invalid or lease expired. Current lock: {current_lock_val}",
            )

        snapshot = self.get_snapshot(session_id)

        # Optimistic Concurrency Control (OCC) check
        if snapshot.version != request.expected_version:
            raise VersionConflictError(
                session_id=session_id,
                expected_version=request.expected_version,
                current_version=snapshot.version,
            )

        contribution = request.contribution

        # Ensure turn index consistency
        if contribution.turn_index == 0 and len(snapshot.contributions) > 0:
            contribution.turn_index = len(snapshot.contributions) + 1

        # Register branch if new
        branch_id = contribution.branch_id
        if branch_id not in snapshot.active_branches:
            branches = snapshot.active_branches
            branches.append(branch_id)
            snapshot.active_branches = branches

        # Register agent if new
        if contribution.agent_id not in snapshot.active_agents:
            snapshot.active_agents.append(contribution.agent_id)

        # Append contribution
        snapshot.contributions.append(contribution)

        # Update active claim on consensus / ratification
        if contribution.tag == PXPTag.RATIFY:
            snapshot.active_claim = contribution.payload.prediction

        # Advance state version and mutation timestamp
        snapshot.version += 1
        snapshot.updated_at = datetime.now(timezone.utc)

        # Save to store
        self._save_snapshot(snapshot)
        self._client.set(self._version_key(session_id), str(snapshot.version))

        # Broadcast mutation event
        self._emit_telemetry(
            session_id=session_id,
            event_type=TelemetryEventType.AGENT_SUBMISSION,
            agent_id=request.agent_id,
            payload={
                "version": snapshot.version,
                "contribution_id": contribution.contribution_id,
                "tag": contribution.tag.value,
                "branch_id": branch_id,
                "turn_index": contribution.turn_index,
                "prediction": contribution.prediction,
            },
        )
        return snapshot

    # --------------------------------------------------------------------------
    # State Inspection & Deadlock Heuristics
    # --------------------------------------------------------------------------

    def get_entries(self, session_id: str, branch_id: str = "main") -> List[AgentContribution]:
        """Returns all contributions for a given branch in chronological order."""
        snapshot = self.get_snapshot(session_id)
        return snapshot.get_branch_entries(branch_id)

    def detect_deadlock(self, session_id: str, window: int = 3, branch_id: str = "main") -> bool:
        """
        Deadlock heuristic: Returns True if the last `window` entries on `branch_id`
        consist exclusively of REFUTE or REJECT tags without resolution.
        """
        entries = self.get_entries(session_id, branch_id)
        if len(entries) < window:
            return False
        recent = entries[-window:]
        return all(e.tag in [PXPTag.REFUTE, PXPTag.REJECT] for e in recent)

    def set_status(
        self,
        session_id: str,
        status: BlackboardStatus | str,
        intelligibility: Optional[IntelligibilityLevel] = None,
    ) -> BlackboardSnapshot:
        """Updates session status and optional intelligibility tier."""
        snapshot = self.get_snapshot(session_id)
        snapshot.status = BlackboardStatus(status) if isinstance(status, str) else status
        if intelligibility is not None:
            snapshot.metadata["intelligibility_level"] = intelligibility.value
        snapshot.updated_at = datetime.now(timezone.utc)
        self._save_snapshot(snapshot)
        return snapshot

    # --------------------------------------------------------------------------
    # Sandbox History Cloning (Student 3 Integration)
    # --------------------------------------------------------------------------

    def clone_sandbox(
        self,
        source_session_id: str,
        target_session_id: str,
        sandbox_branch_id: str,
        up_to_step: Optional[int] = None,
    ) -> BlackboardSnapshot:
        """
        Creates an isolated cloned blackboard session for counterfactual simulation.
        If `up_to_step` is provided, rolls back history to that step.
        Entries are assigned the new `sandbox_branch_id`.
        """
        src = self.get_snapshot(source_session_id)
        new_meta = src.metadata.copy()
        new_meta["active_branches"] = [sandbox_branch_id]
        new_meta["cloned_from"] = source_session_id

        cloned_contributions: List[AgentContribution] = []
        for c in src.contributions:
            if c.branch_id == "main":
                if up_to_step is not None and c.turn_index > up_to_step:
                    continue
                # Clone contribution with updated branch_id
                c_dump = c.model_dump()
                c_dump["metadata"]["branch_id"] = sandbox_branch_id
                c_dump["contribution_id"] = f"{c.contribution_id}_sbx"
                cloned_contributions.append(AgentContribution.model_validate(c_dump))

        sandbox_snapshot = BlackboardSnapshot(
            session_id=target_session_id,
            task_id=src.task_id,
            task_description=src.task_description,
            initial_context=src.initial_context.copy(),
            status=BlackboardStatus.ACTIVE,
            version=len(cloned_contributions),
            active_claim=src.active_claim,
            contributions=cloned_contributions,
            active_agents=src.active_agents.copy(),
            metadata=new_meta,
        )

        self._save_snapshot(sandbox_snapshot)
        self._client.set(self._version_key(target_session_id), str(sandbox_snapshot.version))
        return sandbox_snapshot

    # --------------------------------------------------------------------------
    # Telemetry Streaming
    # --------------------------------------------------------------------------

    def _emit_telemetry(
        self,
        session_id: str,
        event_type: TelemetryEventType,
        payload: Dict[str, Any],
        agent_id: Optional[str] = None,
    ) -> None:
        """Publishes a TelemetryEvent to the Redis channel for Student 4 WebSocket broadcasting."""
        event = TelemetryEvent(
            event_type=event_type,
            session_id=session_id,
            agent_id=agent_id,
            payload=payload,
        )
        channel = self._events_channel(session_id)
        try:
            self._client.publish(channel, event.model_dump_json())
        except Exception as e:
            logger.error("Failed to broadcast telemetry event on %s: %s", channel, e)
