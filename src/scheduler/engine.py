"""
scheduler/engine.py
Deterministic Event-Driven Scheduler for multi-agent turn arbitration.
Orchestrates agent turns, coordinates distributed write locks, detects consensus and deadlocks,
and provides preemption hooks for counterfactual sandbox recovery.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from contracts.schemas import (
    AgentContribution,
    AgentRole,
    BlackboardEntry,
    BlackboardSnapshot,
    BlackboardStatus,
    DeadlockNotification,
    IntelligibilityLevel,
    PEXPayload,
    Prediction,
    Explanation,
    PXPTag,
    ReviseProposal,
    ScheduledTurn,
    SchedulerQueueState,
    StateWriteRequest,
    TelemetryEvent,
    TelemetryEventType,
)
from scheduler.policies import BaseArbitrationPolicy, ReactivePXPPolicy
from scheduler.queue import TurnPriorityQueue

logger = logging.getLogger(__name__)

class SessionBoundBlackboardView:
    """
    Session-bound adapter providing an InMemoryBlackboard-compatible read/write interface
    for agent handlers (like MockPEXAgent and S2 PEXAgent) interacting with RedisBlackboardStore.
    """

    def __init__(self, store: Any, session_id: str):
        self._store = store
        self.session_id = session_id

    def get_entries(self, branch_id: str = "main") -> List[Any]:
        if hasattr(self._store, "get_entries"):
            try:
                return self._store.get_entries(session_id=self.session_id, branch_id=branch_id)
            except TypeError:
                return self._store.get_entries(branch_id=branch_id)
        if hasattr(self._store, "get_snapshot"):
            snap = self._store.get_snapshot(self.session_id)
            return snap.get_branch_entries(branch_id)
        return []

    def get_snapshot(self) -> BlackboardSnapshot:
        if hasattr(self._store, "get_snapshot"):
            try:
                return self._store.get_snapshot(self.session_id)
            except TypeError:
                return self._store.get_snapshot()
        return self._store.get_state()

    def add_entry(self, entry: BlackboardEntry) -> BlackboardEntry:
        # Passthrough: commits are mediated transactionally under lock by the scheduler
        return entry

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)


class DeterministicScheduler:
    """
    Coordinates turn-taking order and execution lifecycle for multi-agent blackboard debates.
    """

    def __init__(
        self,
        session_id: str,
        store: Any,
        policy: Optional[BaseArbitrationPolicy] = None,
        deadlock_window: int = 2,
        consensus_threshold: int = 2,
    ):
        self.session_id: str = session_id
        self.store: Any = store
        self.policy: BaseArbitrationPolicy = policy or ReactivePXPPolicy()
        self.deadlock_window: int = deadlock_window
        self.consensus_threshold: int = consensus_threshold

        self.queue: TurnPriorityQueue = TurnPriorityQueue(session_id=session_id)
        self.registered_agents: Dict[str, Dict[str, Any]] = {}
        self._deadlock_hooks: List[Callable[[DeadlockNotification], Optional[ReviseProposal]]] = []
        self.active_agent_id: Optional[str] = None
        self._last_deadlock_notification: Optional[DeadlockNotification] = None

    # --------------------------------------------------------------------------
    # Agent & Hook Registration
    # --------------------------------------------------------------------------

    def register_agent(
        self,
        agent_id: str,
        agent_role: AgentRole | str = AgentRole.PRIMARY,
        turn_handler: Optional[Any] = None,
        priority: int = 1,
    ) -> None:
        """
        Registers an agent into the scheduler arbitration pool.
        
        Args:
            agent_id: Unique agent identifier.
            agent_role: Domain persona role (PRIMARY, CRITIC, ARBITER, etc.).
            turn_handler: Callable or object with generate_turn(...) interface.
            priority: Base priority score (lower number = higher scheduling priority).
        """
        role_str = agent_role.value if isinstance(agent_role, AgentRole) else str(agent_role)
        self.registered_agents[agent_id] = {
            "role": role_str,
            "handler": turn_handler,
            "priority": priority,
        }

    def register_deadlock_hook(
        self,
        hook: Callable[[DeadlockNotification], Optional[ReviseProposal]],
    ) -> None:
        """Registers a callback invoked when a debate impasse is detected."""
        self._deadlock_hooks.append(hook)

    # --------------------------------------------------------------------------
    # Queue Inspection & State Helpers
    # --------------------------------------------------------------------------

    def get_queue_state(self) -> SchedulerQueueState:
        """Returns the serialized state of the turn arbitration queue."""
        snap = self._get_current_snapshot()
        current_turn = len(snap.contributions)
        return self.queue.get_state(
            current_turn_index=current_turn,
            active_agent_id=self.active_agent_id,
        )

    def _get_current_snapshot(self) -> BlackboardSnapshot:
        """Helper to fetch snapshot regardless of whether store is Redis or InMemory."""
        if hasattr(self.store, "get_snapshot"):
            return self.store.get_snapshot(self.session_id)
        if hasattr(self.store, "get_state"):
            return self.store.get_state()
        raise AttributeError("Underlying store does not support get_snapshot or get_state")

    # --------------------------------------------------------------------------
    # Step-by-Step Turn Execution
    # --------------------------------------------------------------------------

    def step(self) -> Optional[AgentContribution]:
        """
        Executes a single scheduled agent turn:
        1. Checks pause state (e.g. pending deadlock resolution).
        2. Pops next highest-priority ScheduledTurn.
        3. Dispatches turn to agent handler.
        4. Mediates write lock and transactional commit to blackboard.
        5. Checks for terminal conditions (consensus or deadlock loop).
        6. Enqueues subsequent turns via arbitration policy.
        """
        if self.queue.is_paused:
            logger.info("Scheduler is currently paused.")
            return None

        # If queue is empty, request next turns from policy
        if self.queue.is_empty():
            snap = self._get_current_snapshot()
            last_c = snap.contributions[-1] if snap.contributions else None
            initial_turns = self.policy.next_turns(
                session_id=self.session_id,
                current_snapshot=snap,
                registered_agents=self.registered_agents,
                last_contribution=last_c,
            )
            for t in initial_turns:
                self.queue.enqueue(t)

            if self.queue.is_empty():
                logger.info("No turns available to schedule.")
                return None

        # Dequeue highest priority turn
        turn = self.queue.dequeue()
        if turn is None:
            return None

        self.active_agent_id = turn.agent_id

        # Emit TURN_DISPATCHED telemetry
        self._emit_telemetry(
            event_type=TelemetryEventType.TURN_DISPATCHED,
            payload={
                "turn_id": turn.turn_id,
                "agent_id": turn.agent_id,
                "priority": turn.priority,
                "target_contribution_id": turn.target_contribution_id,
            },
            agent_id=turn.agent_id,
        )

        agent_meta = self.registered_agents.get(turn.agent_id)
        if not agent_meta or not agent_meta.get("handler"):
            self.active_agent_id = None
            raise ValueError(f"No executable handler registered for agent '{turn.agent_id}'.")

        handler = agent_meta["handler"]

        # Formulate agent turn
        contrib = self._invoke_agent_handler(handler, turn)

        # Commit contribution to store under lock
        updated_snap = self._commit_turn(turn, contrib)

        # Terminal Condition 1: Check for Consensus
        if self._check_consensus(updated_snap):
            self._set_store_status(BlackboardStatus.CONSENSUS)
            self._emit_telemetry(
                event_type=TelemetryEventType.CONSENSUS_REACHED,
                payload={"active_claim": updated_snap.active_claim.model_dump() if updated_snap.active_claim else None},
            )
            self.active_agent_id = None
            return contrib

        # Terminal Condition 2: Check for Deadlock
        if self._check_deadlock(updated_snap):
            self._handle_deadlock(updated_snap)
            self.active_agent_id = None
            return contrib

        # Debate continues: replenish queue via arbitration policy
        if not self.queue.is_paused and len(self.queue) < len(self.registered_agents):
            next_turns = self.policy.next_turns(
                session_id=self.session_id,
                current_snapshot=updated_snap,
                registered_agents=self.registered_agents,
                last_contribution=contrib,
            )
            for nt in next_turns:
                self.queue.enqueue(nt)

        self.active_agent_id = None
        return contrib

    def _get_bound_blackboard(self) -> Any:
        return SessionBoundBlackboardView(self.store, self.session_id)

    # --------------------------------------------------------------------------
    # Agent Handler Invocation & Commit
    # --------------------------------------------------------------------------

    def _invoke_agent_handler(self, handler: Any, turn: ScheduledTurn) -> AgentContribution:
        """Calls agent handler and normalizes response to an AgentContribution."""
        snap = self._get_current_snapshot()
        current_step = len(snap.contributions) + 1

        # Check if handler has generate_turn (e.g. MockPEXAgent)
        if hasattr(handler, "generate_turn"):
            bound_view = self._get_bound_blackboard()
            raw_entry = handler.generate_turn(
                blackboard=bound_view,
                branch_id="main",
                parent_id=turn.target_contribution_id,
            )
            if isinstance(raw_entry, BlackboardEntry):
                contrib = raw_entry.to_agent_contribution(session_id=self.session_id)
                contrib.turn_index = current_step
                return contrib
            if isinstance(raw_entry, AgentContribution):
                raw_entry.turn_index = current_step
                return raw_entry
            raise TypeError(f"Unexpected return type from generate_turn: {type(raw_entry)}")

        # Check if handler is callable
        if callable(handler):
            res = handler(snap, turn)
            if isinstance(res, AgentContribution):
                if res.turn_index == 0:
                    res.turn_index = current_step
                return res
            if isinstance(res, BlackboardEntry):
                c = res.to_agent_contribution(session_id=self.session_id)
                c.turn_index = current_step
                return c
            raise TypeError(f"Unexpected return type from agent callable: {type(res)}")

        raise TypeError(f"Agent handler for {turn.agent_id} is neither callable nor an agent instance.")

    def _commit_turn(self, turn: ScheduledTurn, contrib: AgentContribution) -> BlackboardSnapshot:
        """Commits the agent turn to the underlying store using distributed write lock."""
        # Case 1: RedisBlackboardStore (Distributed locks + OCC)
        if hasattr(self.store, "lock_context") and hasattr(self.store, "write_state"):
            with self.store.lock_context(self.session_id, turn.agent_id) as lock_token:
                current_snap = self.store.get_snapshot(self.session_id)
                write_req = StateWriteRequest(
                    session_id=self.session_id,
                    agent_id=turn.agent_id,
                    lock_token=lock_token,
                    expected_version=current_snap.version,
                    contribution=contrib,
                )
                return self.store.write_state(write_req)

        # Case 2: InMemoryBlackboard
        if hasattr(self.store, "submit"):
            return self.store.submit(contrib)

        raise NotImplementedError("Storage engine does not support write_state or submit")

    # --------------------------------------------------------------------------
    # Terminal Checks & Deadlock Hooks
    # --------------------------------------------------------------------------

    def _check_consensus(self, snapshot: BlackboardSnapshot) -> bool:
        """
        Consensus criteria:
        Returns True if the trailing `consensus_threshold` contributions across
        distinct agents consist exclusively of RATIFY tags.
        """
        entries = snapshot.get_branch_entries("main")
        if len(entries) < self.consensus_threshold:
            return False

        recent = entries[-self.consensus_threshold:]
        if not all(e.tag == PXPTag.RATIFY for e in recent):
            return False

        distinct_agents = {e.agent_id for e in recent}
        return len(distinct_agents) >= min(self.consensus_threshold, len(self.registered_agents))

    def _check_deadlock(self, snapshot: BlackboardSnapshot) -> bool:
        """
        Deadlock criteria:
        Returns True if the trailing `deadlock_window` entries on main branch
        consist exclusively of REFUTE or REJECT tags across at least 2 distinct agents.
        """
        entries = snapshot.get_branch_entries("main")
        if len(entries) < self.deadlock_window:
            return False

        recent = entries[-self.deadlock_window:]
        refute_or_reject = {PXPTag.REFUTE, PXPTag.REJECT}
        if not all(e.tag in refute_or_reject for e in recent):
            return False

        distinct_agents = {e.agent_id for e in recent}
        return len(distinct_agents) >= 2

    def _handle_deadlock(self, snapshot: BlackboardSnapshot) -> None:
        """Handles a deadlock event: emits notification, pauses queue, and fires hooks."""
        self._set_store_status(BlackboardStatus.DEADLOCK)
        entries = snapshot.get_branch_entries("main")
        recent = entries[-self.deadlock_window:]
        conflicting_agents = list({e.agent_id for e in recent})
        repeated_tags = [e.tag for e in recent]

        notification = DeadlockNotification(
            session_id=self.session_id,
            deadlock_turn_index=len(entries),
            conflicting_agent_ids=conflicting_agents,
            repeated_tags=repeated_tags,
            snapshot=snapshot,
            reason=f"Impasse detected: {len(repeated_tags)} consecutive disagreement turns across {conflicting_agents}.",
        )
        self._last_deadlock_notification = notification

        # Emit telemetry
        self._emit_telemetry(
            event_type=TelemetryEventType.DEADLOCK_DETECTED,
            payload={
                "deadlock_turn_index": notification.deadlock_turn_index,
                "conflicting_agent_ids": conflicting_agents,
                "reason": notification.reason,
            },
        )

        # Pause queue
        self.queue.is_paused = True

        # Invoke registered deadlock hooks
        for hook in self._deadlock_hooks:
            try:
                recovery = hook(notification)
                if recovery:
                    self.inject_counterfactual_recovery(recovery)
                    break
            except Exception as e:
                logger.error("Error executing deadlock hook: %s", e)

    def inject_counterfactual_recovery(self, proposal: ReviseProposal) -> None:
        """
        Injects a winning counterfactual REVISE proposal back into the queue at Priority 0 (highest preemption)
        and unpauses the scheduler to resume execution.
        """
        # Extract agent_id and target_contribution_id from proposal or revised_contribution
        revised = proposal.revised_contribution
        agent_id = getattr(proposal, "proposing_agent_id", None) or revised.agent_id
        target_cid = getattr(proposal, "target_contribution_id", None) or revised.target_contribution_id

        def _recovery_handler(snap: BlackboardSnapshot, t: ScheduledTurn) -> AgentContribution:
            rev_copy = revised.model_copy(deep=True)
            rev_copy.turn_index = len(snap.contributions) + 1
            rev_copy.session_id = self.session_id
            rev_copy.is_counterfactual = True
            return rev_copy

        if agent_id not in self.registered_agents:
            self.register_agent(
                agent_id=agent_id,
                agent_role=AgentRole.COUNTERFACTUAL,
                turn_handler=_recovery_handler,
                priority=0,
            )
        else:
            self.registered_agents[agent_id]["handler"] = _recovery_handler

        # Enqueue recovery turn at Priority 0 (immediate preemption)
        recovery_turn = ScheduledTurn(
            session_id=self.session_id,
            agent_id=agent_id,
            priority=0,
            target_contribution_id=target_cid,
        )
        self.queue.enqueue(recovery_turn)

        # Unpause queue
        self.queue.is_paused = False
        self._set_store_status(BlackboardStatus.ACTIVE)

        logger.info(
            "Injected counterfactual REVISE recovery proposal from %s at Priority 0. Scheduler unpaused.",
            agent_id,
        )

    # --------------------------------------------------------------------------
    # Session Run Loop
    # --------------------------------------------------------------------------

    def run(self, max_turns: int = 20) -> BlackboardSnapshot:
        """
        Runs the debate session loop until a terminal condition is met:
        - CONSENSUS reached
        - DEADLOCK detected (without immediate counterfactual recovery)
        - Queue empty and no turns can be scheduled
        - max_turns ceiling reached
        """
        turns_executed = 0
        while turns_executed < max_turns:
            snap = self._get_current_snapshot()
            if snap.status in [BlackboardStatus.CONSENSUS, BlackboardStatus.RESOLVED]:
                break
            if snap.status == BlackboardStatus.DEADLOCK and self.queue.is_paused:
                break

            contrib = self.step()
            if contrib is None:
                break
            turns_executed += 1

        return self._get_current_snapshot()

    def _set_store_status(self, status: BlackboardStatus) -> None:
        """Helper to update status in store."""
        if hasattr(self.store, "set_status"):
            self.store.set_status(self.session_id, status) if hasattr(self.store, "create_session") else self.store.set_status(status.value)

    def _emit_telemetry(
        self,
        event_type: TelemetryEventType,
        payload: Dict[str, Any],
        agent_id: Optional[str] = None,
    ) -> None:
        """Emits telemetry events if store provides _emit_telemetry."""
        if hasattr(self.store, "_emit_telemetry"):
            self.store._emit_telemetry(
                session_id=self.session_id,
                event_type=event_type,
                payload=payload,
                agent_id=agent_id,
            )
