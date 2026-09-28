"""
blackboard/store.py
Thread-safe, in-memory Directed Acyclic Graph (DAG) state store for the Blackboard.
Provides deep-copy and sandbox branching capabilities needed for Student 3's counterfactual engine.
"""

import copy
import threading
from typing import List, Optional
from contracts.schemas import (
    AgentContribution,
    BlackboardEntry,
    BlackboardSnapshot,
    BlackboardState,
    IntelligibilityLevel,
    PXPTag,
)


class InMemoryBlackboard:
    """Thread-safe, in-memory blackboard data store."""

    def __init__(
        self,
        session_id: str,
        task_id: str,
        problem_statement: str,
        ground_truth: Optional[str] = None,
    ):
        self._lock = threading.RLock()
        self._state = BlackboardState(
            session_id=session_id,
            task_id=task_id,
            problem_statement=problem_statement,
            ground_truth=ground_truth,
            entries=[],
            active_branches=["main"],
            status="IN_PROGRESS",
            intelligibility_level=IntelligibilityLevel.NONE,
        )

    def add_entry(self, entry: BlackboardEntry) -> BlackboardEntry:
        """Appends a new entry to the blackboard in a thread-safe manner."""
        with self._lock:
            if entry.branch_id not in self._state.active_branches:
                self._state.active_branches.append(entry.branch_id)
            self._state.entries.append(entry)
            return entry

    def get_entries(self, branch_id: str = "main") -> List[BlackboardEntry]:
        """Returns a copy of all entries for a specific branch."""
        with self._lock:
            return [copy.deepcopy(e) for e in self._state.entries if e.branch_id == branch_id]

    def get_state(self) -> BlackboardState:
        """Returns a snapshot copy of the full blackboard state."""
        with self._lock:
            return copy.deepcopy(self._state)

    def get_snapshot(self, session_id: Optional[str] = None) -> BlackboardSnapshot:
        """Implements Student 2 BoardClient.get_snapshot Protocol."""
        with self._lock:
            return copy.deepcopy(self._state)

    def submit(self, contribution: AgentContribution) -> BlackboardSnapshot:
        """Implements Student 2 BoardClient.submit Protocol."""
        with self._lock:
            entry = BlackboardEntry.from_agent_contribution(contribution)
            self.add_entry(entry)
            return copy.deepcopy(self._state)

    def clone_sandbox(
        self, 
        sandbox_branch_id: str, 
        up_to_step: Optional[int] = None
    ) -> "InMemoryBlackboard":
        """
        Creates an isolated in-memory clone of this blackboard for counterfactual simulation.
        If `up_to_step` is provided, rolls back history to that step.
        Entries are assigned the new `sandbox_branch_id`.
        """
        with self._lock:
            cloned = InMemoryBlackboard(
                session_id=f"{self._state.session_id}_sandbox",
                task_id=self._state.task_id,
                problem_statement=self._state.problem_statement,
                ground_truth=self._state.ground_truth,
            )
            cloned._state.active_branches = [sandbox_branch_id]

            # Copy entries up to up_to_step
            for entry in self._state.entries:
                if entry.branch_id == "main":
                    if up_to_step is not None and entry.step_number > up_to_step:
                        continue
                    copied_entry = copy.deepcopy(entry)
                    copied_entry.branch_id = sandbox_branch_id
                    cloned._state.entries.append(copied_entry)

            return cloned

    def detect_deadlock(self, window: int = 3) -> bool:
        """
        Deadlock heuristic: Returns True if the last `window` entries on the main branch
        consist exclusively of REFUTE or REJECT tags without any resolution.
        """
        with self._lock:
            main_entries = [e for e in self._state.entries if e.branch_id == "main"]
            if len(main_entries) < window:
                return False
            recent = main_entries[-window:]
            return all(e.tag in [PXPTag.REFUTE, PXPTag.REJECT] for e in recent)

    def set_status(self, status: str, intelligibility: Optional[IntelligibilityLevel] = None):
        """Updates session status and intelligibility level."""
        with self._lock:
            self._state.status = status
            if intelligibility is not None:
                self._state.intelligibility_level = intelligibility
