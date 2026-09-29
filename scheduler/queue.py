"""
scheduler/queue.py
Thread-safe Priority Queue for agent turn arbitration in the Intelligible Blackboard Architecture.
Orders ScheduledTurn instances strictly by (priority, scheduled_at), where lower priority numbers
represent higher precedence (Priority 0 is reserved for Arbiter and Counterfactual preemption).
"""

from __future__ import annotations

import heapq
import threading
from typing import Any, List, Optional

from contracts.schemas import ScheduledTurn, SchedulerQueueState


class _PriorityTurnWrapper:
    """Wrapper ensuring deterministic sorting for heapq comparisons."""

    def __init__(self, turn: ScheduledTurn):
        self.turn = turn

    def __lt__(self, other: _PriorityTurnWrapper) -> bool:
        if self.turn.priority != other.turn.priority:
            return self.turn.priority < other.turn.priority
        # Tie-breaker: earlier scheduled timestamp
        return self.turn.scheduled_at < other.turn.scheduled_at

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, _PriorityTurnWrapper):
            return False
        return self.turn.turn_id == other.turn.turn_id


class TurnPriorityQueue:
    """
    Thread-safe turn queue maintaining pending agent turns sorted by priority.
    """

    def __init__(self, session_id: str):
        self.session_id: str = session_id
        self._heap: List[_PriorityTurnWrapper] = []
        self._lock = threading.RLock()
        self._is_paused: bool = False

    @property
    def is_paused(self) -> bool:
        with self._lock:
            return self._is_paused

    @is_paused.setter
    def is_paused(self, value: bool) -> None:
        with self._lock:
            self._is_paused = value

    def enqueue(self, turn: ScheduledTurn) -> None:
        """Pushes a ScheduledTurn into the priority queue."""
        with self._lock:
            heapq.heappush(self._heap, _PriorityTurnWrapper(turn))

    def dequeue(self) -> Optional[ScheduledTurn]:
        """Pops and returns the highest priority turn, or None if empty."""
        with self._lock:
            if not self._heap:
                return None
            wrapper = heapq.heappop(self._heap)
            return wrapper.turn

    def peek(self) -> Optional[ScheduledTurn]:
        """Returns the highest priority turn without removing it, or None if empty."""
        with self._lock:
            if not self._heap:
                return None
            return self._heap[0].turn

    def clear(self) -> None:
        """Empties the priority queue."""
        with self._lock:
            self._heap.clear()

    def remove_by_agent(self, agent_id: str) -> int:
        """Removes all queued turns assigned to agent_id, returning count removed."""
        with self._lock:
            initial_count = len(self._heap)
            self._heap = [w for w in self._heap if w.turn.agent_id != agent_id]
            heapq.heapify(self._heap)
            return initial_count - len(self._heap)

    def get_pending_turns(self) -> List[ScheduledTurn]:
        """Returns a copy of all pending turns in priority order."""
        with self._lock:
            sorted_wrappers = sorted(self._heap)
            return [w.turn.model_copy() for w in sorted_wrappers]

    def get_state(
        self,
        current_turn_index: int,
        active_agent_id: Optional[str] = None,
    ) -> SchedulerQueueState:
        """Builds a SchedulerQueueState contract representation of the current queue."""
        with self._lock:
            return SchedulerQueueState(
                session_id=self.session_id,
                current_turn_index=current_turn_index,
                active_agent_id=active_agent_id,
                pending_queue=self.get_pending_turns(),
                is_paused=self._is_paused,
            )

    def __len__(self) -> int:
        with self._lock:
            return len(self._heap)

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._heap) == 0
