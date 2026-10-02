"""Interface S2 agents use to talk to S1's blackboard, plus an in-memory stub.

S1's real store should satisfy `BoardClient` (or be wrapped by a thin adapter).
Until it lands, `InMemoryBoard` lets agents run end-to-end (per plan: build against stubs).
"""

from __future__ import annotations

import copy
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Protocol, runtime_checkable
from uuid import uuid4

from contracts.schemas import (
    AgentContribution,
    BlackboardSnapshot,
    BlackboardStatus,
    LockAcquireResponse,
    PXPTag,
)


@runtime_checkable
class BoardClient(Protocol):
    def get_snapshot(self, session_id: str) -> BlackboardSnapshot: ...

    def submit(self, contribution: AgentContribution) -> BlackboardSnapshot:
        """Validate + append atomically; return the new snapshot. Raise ValueError on rejection."""
        ...


class StaleTurnError(ValueError):
    """The board moved on since the agent read it (turn_index mismatch)."""


class InMemoryBoard:
    """Thread-safe single-process stand-in for S1's blackboard (dev/tests only)."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: Dict[str, BlackboardSnapshot] = {}

    def create_session(
        self,
        task_id: str,
        task_description: str,
        initial_context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> BlackboardSnapshot:
        with self._lock:
            sid = session_id or str(uuid4())
            snap = BlackboardSnapshot(
                session_id=sid,
                task_id=task_id,
                task_description=task_description,
                initial_context=initial_context or {},
                status=BlackboardStatus.ACTIVE,
            )
            self._sessions[sid] = snap
            return snap.model_copy(deep=True)

    def get_snapshot(self, session_id: str) -> BlackboardSnapshot:
        with self._lock:
            return self._sessions[session_id].model_copy(deep=True)

    def submit(self, contribution: AgentContribution) -> BlackboardSnapshot:
        with self._lock:
            snap = self._sessions[contribution.session_id]
            if contribution.turn_index != len(snap.contributions):
                raise StaleTurnError(
                    f"turn_index {contribution.turn_index} != expected {len(snap.contributions)}"
                )
            ids = {c.contribution_id for c in snap.contributions}
            if contribution.tag != PXPTag.REVISE and contribution.target_contribution_id not in ids:
                raise ValueError(f"{contribution.tag.value} must target an existing contribution")
            snap.contributions.append(copy.deepcopy(contribution))
            if contribution.tag in (PXPTag.REVISE, PXPTag.RATIFY):
                snap.active_claim = contribution.payload.prediction
            snap.version += 1
            snap.updated_at = datetime.now(timezone.utc)
            if contribution.agent_id not in snap.active_agents:
                snap.active_agents.append(contribution.agent_id)
            return snap.model_copy(deep=True)

    # stand-in lock API so agents can be exercised against S1-style leasing
    def acquire(self, session_id: str, agent_id: str) -> LockAcquireResponse:
        return LockAcquireResponse(acquired=True, lock_token=str(uuid4()))
