"""
sandbox/manager.py
Isolated Sandbox & Branch Manager for the Intelligible Blackboard Architecture.
Provides isolated, thread-safe branch sessions, hierarchical checkpoint backtracking,
and telemetry graph export for counterfactual debate exploration.
"""

import copy
import threading
import uuid
from typing import Any, Dict, List, Optional

from contracts.schemas import BlackboardEntry, BlackboardState, PXPTag
from blackboard.store import InMemoryBlackboard
from counterfactual.attribution import AttributionResult, DeadlockAttributionEngine


class SandboxSession:
    """
    Isolated counterfactual debate branch session.
    Wraps an isolated InMemoryBlackboard, tracks simulated entries,
    and supports hierarchical checkpoint backtracking up to 3 levels.
    """

    def __init__(
        self,
        session_id: str,
        source_session_id: str,
        sandbox_branch_id: str,
        board: InMemoryBlackboard,
        start_step: int,
        rollback_step: int,
        checkpoints_stack: Optional[List[BlackboardEntry]] = None,
        source_branch_id: str = "main",
        source_board: Optional[InMemoryBlackboard] = None,
        current_checkpoint_level: int = 0,
        escalation_exhausted: bool = False,
        simulated_entries: Optional[List[BlackboardEntry]] = None,
    ):
        self.session_id: str = session_id
        self.source_session_id: str = source_session_id
        self.source_branch_id: str = source_branch_id
        self.sandbox_branch_id: str = sandbox_branch_id
        self.board: InMemoryBlackboard = board
        self.start_step: int = start_step
        self.rollback_step: int = rollback_step
        self.checkpoints_stack: List[BlackboardEntry] = (
            checkpoints_stack if checkpoints_stack is not None else []
        )
        self.current_checkpoint_level: int = current_checkpoint_level
        self.escalation_exhausted: bool = escalation_exhausted
        self.simulated_entries: List[BlackboardEntry] = (
            simulated_entries if simulated_entries is not None else []
        )
        self._source_board: Optional[InMemoryBlackboard] = source_board
        self._lock = threading.RLock()

    def add_simulated_entry(self, entry: BlackboardEntry) -> BlackboardEntry:
        """
        Appends a new simulated entry to the sandbox with branch_id = self.sandbox_branch_id.
        Thread-safe and isolated from the live board.
        """
        with self._lock:
            entry_copy = copy.deepcopy(entry)
            entry_copy.branch_id = self.sandbox_branch_id

            added = self.board.add_entry(entry_copy)
            self.simulated_entries.append(added)
            return copy.deepcopy(added)

    def get_simulated_entries(self) -> List[BlackboardEntry]:
        """
        Returns only entries created during simulation.
        """
        with self._lock:
            return [copy.deepcopy(e) for e in self.simulated_entries]

    def get_all_entries(self) -> List[BlackboardEntry]:
        """
        Returns cloned history + simulated entries on this sandbox branch in chronological order.
        """
        with self._lock:
            entries = self.board.get_entries(self.sandbox_branch_id)
            return sorted(entries, key=lambda e: e.step_number)

    def escalate_to_prior_checkpoint(self) -> bool:
        """
        Moves back one checkpoint in checkpoints_stack.
        Max 3 levels (level 0 = CN, level 1 = CN-1, level 2 = CN-2).
        If level >= len(checkpoints_stack) or level >= 3,
        sets escalation_exhausted = True and returns False.
        """
        with self._lock:
            if self.escalation_exhausted:
                return False

            next_level = self.current_checkpoint_level + 1
            if next_level >= len(self.checkpoints_stack) or next_level >= 3:
                self.escalation_exhausted = True
                return False

            self.current_checkpoint_level = next_level
            target_cp = self.checkpoints_stack[self.current_checkpoint_level]
            self.start_step = min(target_cp.step_number, self.rollback_step)

            # If source board is available, reload the sandbox window to the new checkpoint
            if self._source_board is not None:
                self._reload_board_history()

            return True

    def _reload_board_history(self) -> None:
        """Reloads the sandbox board with entries from the new start_step to rollback_step."""
        with self._lock:
            if self._source_board is None:
                return

            source_entries = self._source_board.get_entries(branch_id=self.source_branch_id)
            scoped_entries: List[BlackboardEntry] = []
            for e in source_entries:
                if self.start_step <= e.step_number <= self.rollback_step:
                    cp_e = copy.deepcopy(e)
                    cp_e.branch_id = self.sandbox_branch_id
                    scoped_entries.append(cp_e)

            with self.board._lock:
                self.board._state.entries = scoped_entries
                self.board._state.active_branches = [self.sandbox_branch_id]
                self.simulated_entries = []

    def export_graph_for_telemetry(self) -> Dict[str, Any]:
        """
        Formats nodes and edges for Student 4's React Flow split-screen UI.
        Includes id, label, tag, agent, parent_id, is_simulated, position.
        Prevents dangling edges where parent_id is outside the scoped window.
        """
        with self._lock:
            all_entries = self.get_all_entries()
            sim_ids = {e.entry_id for e in self.simulated_entries}
            node_ids = {e.entry_id for e in all_entries}

            nodes: List[Dict[str, Any]] = []
            edges: List[Dict[str, Any]] = []

            for entry in all_entries:
                is_sim = entry.entry_id in sim_ids
                tag_str = entry.tag.value if hasattr(entry.tag, "value") else str(entry.tag)
                label = f"[{tag_str}] {entry.agent_id}: {entry.prediction}"

                node_dict = {
                    "id": entry.entry_id,
                    "label": label,
                    "tag": tag_str,
                    "agent": entry.agent_id,
                    "agent_role": entry.agent_role,
                    "parent_id": entry.parent_id,
                    "is_simulated": is_sim,
                    "step_number": entry.step_number,
                    "prediction": entry.prediction,
                    "explanation": entry.explanation,
                    "confidence": entry.confidence,
                    "position": {"x": 0.0, "y": 0.0},
                    "data": {
                        "id": entry.entry_id,
                        "label": label,
                        "tag": tag_str,
                        "agent": entry.agent_id,
                        "agent_role": entry.agent_role,
                        "parent_id": entry.parent_id,
                        "is_simulated": is_sim,
                        "prediction": entry.prediction,
                        "explanation": entry.explanation,
                        "confidence": entry.confidence,
                    },
                }
                nodes.append(node_dict)

                # Only emit edges if the parent exists in the scoped window node set
                if entry.parent_id and entry.parent_id in node_ids:
                    edges.append({
                        "id": f"edge_{entry.parent_id}_{entry.entry_id}",
                        "source": entry.parent_id,
                        "target": entry.entry_id,
                        "is_simulated": is_sim,
                    })

            return {
                "session_id": self.session_id,
                "sandbox_branch_id": self.sandbox_branch_id,
                "start_step": self.start_step,
                "rollback_step": self.rollback_step,
                "current_checkpoint_level": self.current_checkpoint_level,
                "escalation_exhausted": self.escalation_exhausted,
                "nodes": nodes,
                "edges": edges,
            }


class SandboxManager:
    """
    Factory and registry for SandboxSessions.
    Extracts consensus checkpoints, creates isolated sandbox branches,
    and manages session lifecycle.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._sessions: Dict[str, SandboxSession] = {}

    def create_session_from_attribution(
        self,
        live_board: InMemoryBlackboard,
        attribution: AttributionResult,
        sandbox_branch_id: Optional[str] = None,
    ) -> SandboxSession:
        """
        Factory extracting checkpoints, creating unique branch sandbox_cf_<uuid>,
        cloning only the scoped window (start_step to rollback_step),
        and registering the session.
        """
        with self._lock:
            live_state = live_board.get_state()
            live_entries = live_board.get_entries(branch_id="main")

            # Extract consensus checkpoints from live board
            engine = DeadlockAttributionEngine()
            all_cps = engine.find_consensus_checkpoints(live_entries)

            # Determine rollback step from attribution
            rollback_step = attribution.rollback_step
            if rollback_step <= 0 and attribution.divergence_entry_id:
                for e in live_entries:
                    if e.entry_id == attribution.divergence_entry_id:
                        rollback_step = e.step_number
                        break
            if rollback_step <= 0 and live_entries:
                rollback_step = live_entries[-1].step_number

            # Checkpoints stack: up to 3 consensus checkpoints in reverse chronological order
            # (CN = level 0, CN-1 = level 1, CN-2 = level 2)
            cps_prior = (
                [cp for cp in all_cps if cp.step_number <= rollback_step]
                if rollback_step > 0
                else all_cps
            )
            chosen_cps = cps_prior if cps_prior else all_cps
            checkpoints_stack: List[BlackboardEntry] = list(reversed(chosen_cps))[:3]

            # If find_consensus_checkpoints missed any explicit attribution checkpoint IDs
            if not checkpoints_stack:
                if attribution.latest_checkpoint_id:
                    for e in live_entries:
                        if e.entry_id == attribution.latest_checkpoint_id:
                            checkpoints_stack.append(e)
                            break
                if attribution.fallback_checkpoint_id:
                    for e in live_entries:
                        if e.entry_id == attribution.fallback_checkpoint_id:
                            if not any(cp.entry_id == e.entry_id for cp in checkpoints_stack):
                                checkpoints_stack.append(e)
                            break

            # Determine start_step
            if checkpoints_stack:
                start_step = checkpoints_stack[0].step_number
            elif attribution.latest_checkpoint_step > 0:
                start_step = attribution.latest_checkpoint_step
            else:
                start_step = 1

            if rollback_step < start_step:
                start_step = rollback_step

            # Generate unique branch ID and session ID
            if not sandbox_branch_id:
                branch_uuid = uuid.uuid4().hex[:8]
                sandbox_branch_id = f"sandbox_cf_{branch_uuid}"

            session_id = f"sess_{sandbox_branch_id}"

            # Clone scoped window: start_step to rollback_step
            scoped_entries: List[BlackboardEntry] = []
            for entry in live_entries:
                if start_step <= entry.step_number <= rollback_step:
                    copied = copy.deepcopy(entry)
                    copied.branch_id = sandbox_branch_id
                    scoped_entries.append(copied)

            # Build isolated board
            sandbox_board = InMemoryBlackboard(
                session_id=f"{live_state.session_id}_{sandbox_branch_id}",
                task_id=live_state.task_id,
                problem_statement=live_state.problem_statement,
                ground_truth=live_state.ground_truth,
            )
            sandbox_board._state.active_branches = [sandbox_branch_id]
            sandbox_board._state.entries = scoped_entries

            session = SandboxSession(
                session_id=session_id,
                source_session_id=live_state.session_id,
                source_branch_id="main",
                sandbox_branch_id=sandbox_branch_id,
                board=sandbox_board,
                start_step=start_step,
                rollback_step=rollback_step,
                checkpoints_stack=checkpoints_stack,
                source_board=live_board,
                current_checkpoint_level=0,
                escalation_exhausted=False,
                simulated_entries=[],
            )

            self._sessions[session_id] = session
            return session

    def get_session(self, session_id: str) -> Optional[SandboxSession]:
        """Retrieves an active sandbox session by session_id or sandbox_branch_id."""
        with self._lock:
            if session_id in self._sessions:
                return self._sessions[session_id]
            for s in self._sessions.values():
                if s.sandbox_branch_id == session_id:
                    return s
            return None

    def close_session(self, session_id: str) -> bool:
        """Closes and unregisters a sandbox session."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            for sid, s in list(self._sessions.items()):
                if s.sandbox_branch_id == session_id:
                    del self._sessions[sid]
                    return True
            return False

    def cleanup_all(self) -> None:
        """Cleans up and removes all tracked sandbox sessions."""
        with self._lock:
            self._sessions.clear()
