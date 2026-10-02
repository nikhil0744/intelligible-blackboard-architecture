"""
counterfactual/attribution.py
Deadlock Attribution and Divergence Engine for the Intelligible Blackboard Architecture.
Identifies deadlock loops, locates divergence trigger nodes by climbing parent_id DAG links,
and pinpoints consensus checkpoints for counterfactual sandbox rollbacks.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pydantic import BaseModel, Field

from contracts.schemas import BlackboardEntry, BlackboardState, PXPTag


class AttributionResult(BaseModel):
    """
    Diagnostic result returned by the DeadlockAttributionEngine.
    Encapsulates deadlock detection, conflicting parties, divergence nodes,
    rollback targets, consensus checkpoints, and clashing evidence.
    """
    deadlock_detected: bool = Field(
        default=False, 
        description="True if an unresolvable REFUTE/REJECT loop is detected"
    )
    conflicting_agents: List[str] = Field(
        default_factory=list, 
        description="IDs of agents participating in the deadlock loop"
    )
    conflict_loop_entries: List[str] = Field(
        default_factory=list, 
        description="Entry IDs composing the unresolved conflict sequence"
    )
    divergence_entry_id: Optional[str] = Field(
        default=None, 
        description="ID of the entry that triggered the divergence"
    )
    rollback_step: int = Field(
        default=0, 
        description="Sequential step number of the divergence node for sandbox rollback"
    )
    latest_checkpoint_id: Optional[str] = Field(
        default=None, 
        description="ID of the most recent ratified consensus checkpoint"
    )
    latest_checkpoint_step: int = Field(
        default=0, 
        description="Step number of the most recent ratified consensus checkpoint"
    )
    fallback_checkpoint_id: Optional[str] = Field(
        default=None, 
        description="ID of the penultimate ratified consensus checkpoint"
    )
    fallback_checkpoint_step: int = Field(
        default=0, 
        description="Step number of the penultimate ratified consensus checkpoint"
    )
    contested_predictions: Dict[str, str] = Field(
        default_factory=dict, 
        description="Mapping of conflicting agent_id to their contested prediction string"
    )
    clashing_evidence: List[str] = Field(
        default_factory=list, 
        description="Deduplicated list of evidence references cited in the dispute"
    )
    attribution_confidence: float = Field(
        default=0.0, 
        ge=0.0, 
        le=1.0, 
        description="Confidence score of the attribution diagnostic"
    )
    reason: str = Field(
        default="", 
        description="Human/LLM-readable diagnostic explanation of the deadlock state"
    )


class DeadlockAttributionEngine:
    """
    Diagnostic engine analyzing blackboard debate histories for deadlocks,
    tracing causal divergence origins, and extracting consensus checkpoints.
    """

    def is_deadlocked(self, entries: List[BlackboardEntry], window: int = 2) -> bool:
        """
        Deadlock heuristic: Returns True if the trailing `window` entries
        consist exclusively of REFUTE or REJECT tags across at least two distinct conflicting agents.
        """
        if not entries or window < 2 or len(entries) < window:
            return False

        recent = entries[-window:]
        refute_or_reject = {
            PXPTag.REFUTE,
            PXPTag.REJECT,
            PXPTag.REFUTE.value,
            PXPTag.REJECT.value,
        }

        if not all(e.tag in refute_or_reject for e in recent):
            return False

        distinct_agents = {e.agent_id for e in recent}
        return len(distinct_agents) >= 2

    def _get_trailing_conflict_entries(
        self, entries: List[BlackboardEntry]
    ) -> List[BlackboardEntry]:
        """Extracts the contiguous sequence of trailing REFUTE/REJECT entries."""
        refute_or_reject = {
            PXPTag.REFUTE,
            PXPTag.REJECT,
            PXPTag.REFUTE.value,
            PXPTag.REJECT.value,
        }
        conflict_entries: List[BlackboardEntry] = []
        for e in reversed(entries):
            if e.tag in refute_or_reject:
                conflict_entries.append(e)
            else:
                break
        conflict_entries.reverse()
        return conflict_entries

    def find_consensus_checkpoints(self, entries: List[BlackboardEntry]) -> List[BlackboardEntry]:
        """
        Finds ratified consensus checkpoints in chronological order.
        Identifies entries with tag RATIFY or explicit checkpoint metadata.
        """
        checkpoints: List[BlackboardEntry] = []
        for e in entries:
            if (
                e.tag == PXPTag.RATIFY
                or e.metadata.get("is_checkpoint") is True
                or e.metadata.get("is_consensus_checkpoint") is True
                or e.metadata.get("consensus") is True
            ):
                checkpoints.append(e)
        return checkpoints

    def locate_divergence_node(
        self, entries: List[BlackboardEntry]
    ) -> Tuple[Optional[str], int, List[str]]:
        """
        Climbs parent_id links from first polarized rejection back to the trigger node.
        Includes cycle detection to prevent infinite loops on malformed parent graphs.
        
        Returns:
            Tuple[Optional[str], int, List[str]]:
                - divergence_entry_id: ID of the proposal/revision triggering the dispute
                - rollback_step: Step number of the trigger node
                - climb_path: Sequence of entry IDs traversed from rejection to trigger node
        """
        if not entries:
            return (None, 0, [])

        refute_or_reject = {
            PXPTag.REFUTE,
            PXPTag.REJECT,
            PXPTag.REFUTE.value,
            PXPTag.REJECT.value,
        }
        entry_map: Dict[str, BlackboardEntry] = {e.entry_id: e for e in entries}

        conflict_entries = self._get_trailing_conflict_entries(entries)
        if not conflict_entries:
            return (None, 0, [])

        first_rejection = conflict_entries[0]
        climb_path: List[str] = [first_rejection.entry_id]

        curr = first_rejection
        trigger_entry: Optional[BlackboardEntry] = None
        visited: Set[str] = {curr.entry_id}

        while curr.parent_id is not None:
            # Cycle detection guard
            if curr.parent_id in visited:
                break
            visited.add(curr.parent_id)

            parent = entry_map.get(curr.parent_id)
            if not parent:
                climb_path.append(curr.parent_id)
                return (curr.parent_id, 0, climb_path)

            climb_path.append(parent.entry_id)
            if parent.tag not in refute_or_reject:
                trigger_entry = parent
                break
            curr = parent

        if trigger_entry is None and climb_path:
            root_id = climb_path[-1]
            trigger_entry = entry_map.get(root_id)

        if trigger_entry:
            return (trigger_entry.entry_id, trigger_entry.step_number, climb_path)

        return (None, 0, climb_path)

    def analyze(
        self, 
        board_or_entries: Any,
        window: int = 2,
        branch_id: str = "main",
    ) -> AttributionResult:
        """
        Master diagnostic function analyzing a blackboard debate session.
        Accepts InMemoryBlackboard, BlackboardState, or List[BlackboardEntry].
        Filters entries by branch_id (defaults to 'main').
        """
        entries = self._extract_entries(board_or_entries, branch_id=branch_id)

        if not entries:
            return AttributionResult(
                deadlock_detected=False,
                reason="No entries found in blackboard session."
            )

        # Check for ratified consensus checkpoints
        checkpoints = self.find_consensus_checkpoints(entries)
        latest_cp = checkpoints[-1] if checkpoints else None
        fallback_cp = checkpoints[-2] if len(checkpoints) >= 2 else None

        latest_cp_id = latest_cp.entry_id if latest_cp else None
        latest_cp_step = latest_cp.step_number if latest_cp else 0
        fallback_cp_id = fallback_cp.entry_id if fallback_cp else None
        fallback_cp_step = fallback_cp.step_number if fallback_cp else 0

        # Check if deadlocked
        deadlocked = self.is_deadlocked(entries, window=window)

        if not deadlocked:
            return AttributionResult(
                deadlock_detected=False,
                latest_checkpoint_id=latest_cp_id,
                latest_checkpoint_step=latest_cp_step,
                fallback_checkpoint_id=fallback_cp_id,
                fallback_checkpoint_step=fallback_cp_step,
                reason="No deadlock detected; debate is progressing constructively or reached consensus."
            )

        # Extract trailing conflict entries
        conflict_entries = self._get_trailing_conflict_entries(entries)

        # Locate divergence trigger
        div_id, rollback_step, _ = self.locate_divergence_node(entries)

        # Extract conflicting agent roster (preserving appearance order)
        conflicting_agents: List[str] = []
        seen_agents = set()
        for e in conflict_entries:
            if e.agent_id not in seen_agents:
                seen_agents.add(e.agent_id)
                conflicting_agents.append(e.agent_id)

        # Contested predictions mapping
        contested_predictions: Dict[str, str] = {}
        for e in conflict_entries:
            contested_predictions[e.agent_id] = e.prediction

        # Clashing evidence aggregation
        clashing_evidence: List[str] = []
        seen_evidence = set()
        trigger_entry = next((e for e in entries if e.entry_id == div_id), None)
        all_relevant = list(conflict_entries) + ([trigger_entry] if trigger_entry else [])
        for e in all_relevant:
            for ref in e.evidence_refs:
                if ref not in seen_evidence:
                    seen_evidence.add(ref)
                    clashing_evidence.append(ref)

        # Attribution confidence based on conflict entries certainty
        avg_conf = (
            sum(e.confidence for e in conflict_entries) / len(conflict_entries)
            if conflict_entries else 1.0
        )
        attribution_confidence = round(float(avg_conf), 2)

        reason = (
            f"Deadlock detected: Circular dispute among agents {conflicting_agents} "
            f"around divergence at entry '{div_id}' (step {rollback_step})."
        )

        return AttributionResult(
            deadlock_detected=True,
            conflicting_agents=conflicting_agents,
            conflict_loop_entries=[e.entry_id for e in conflict_entries],
            divergence_entry_id=div_id,
            rollback_step=rollback_step,
            latest_checkpoint_id=latest_cp_id,
            latest_checkpoint_step=latest_cp_step,
            fallback_checkpoint_id=fallback_cp_id,
            fallback_checkpoint_step=fallback_cp_step,
            contested_predictions=contested_predictions,
            clashing_evidence=clashing_evidence,
            attribution_confidence=attribution_confidence,
            reason=reason,
        )

    def _extract_entries(
        self, board_or_entries: Any, branch_id: str = "main"
    ) -> List[BlackboardEntry]:
        """Polymorphic helper extracting entry list from various inputs, filtered by branch_id."""
        if isinstance(board_or_entries, list):
            if board_or_entries and isinstance(board_or_entries[0], dict):
                entries = [BlackboardEntry.model_validate(e) for e in board_or_entries]
            else:
                entries = list(board_or_entries)
            if branch_id is not None:
                return [e for e in entries if getattr(e, "branch_id", "main") == branch_id]
            return entries

        if hasattr(board_or_entries, "get_entries"):
            return board_or_entries.get_entries(branch_id=branch_id)

        if hasattr(board_or_entries, "get_branch_entries"):
            return board_or_entries.get_branch_entries(branch_id=branch_id)

        if hasattr(board_or_entries, "entries"):
            return [e for e in board_or_entries.entries if getattr(e, "branch_id", "main") == branch_id]

        if isinstance(board_or_entries, dict):
            state = BlackboardState.model_validate(board_or_entries)
            return state.get_branch_entries(branch_id=branch_id)

        raise TypeError(f"Unsupported board or entries type: {type(board_or_entries)}")
