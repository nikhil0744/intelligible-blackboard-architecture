"""
counterfactual/solver.py
Task 5: Live Deadlock Solver Daemon & Preemption Bridge.

Integrates Student 1's DeterministicScheduler deadlock notification hook with
Student 3's DeadlockAttributionEngine, SandboxManager, CounterfactualSimulator,
and CreditAssignmentScorer into an autonomous self-healing recovery daemon.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Callable, Dict, List, Optional
import uuid

from pydantic import BaseModel, ConfigDict, Field

from contracts.schemas import (
    AgentContribution,
    AgentRole,
    BlackboardEntry,
    BlackboardSnapshot,
    BlackboardStatus,
    CounterfactualBranch,
    CreditAssignmentDelta,
    DeadlockNotification,
    Explanation,
    PEXPayload,
    Prediction,
    PXPTag,
    ReviseProposal,
    TelemetryEvent,
    TelemetryEventType,
)
from counterfactual.attribution import AttributionResult, DeadlockAttributionEngine
from counterfactual.scoring import BranchScoreDetails, CreditAssignmentScorer
from counterfactual.simulator import CounterfactualSimulator, SimulationResult, SimulationStatus
from blackboard.store import InMemoryBlackboard
from sandbox.manager import SandboxManager, SandboxSession

logger = logging.getLogger(__name__)


class SolverConfig(BaseModel):
    """Configuration parameters for DeadlockSolverDaemon."""
    model_config = ConfigDict(extra="ignore")

    convergence_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Minimum composite score required to declare counterfactual recovery successful.",
    )
    max_simulation_rounds: int = Field(
        default=5,
        ge=1,
        description="Maximum hill-climbing search iterations per sandbox session.",
    )
    max_escalation_levels: int = Field(
        default=3,
        ge=1,
        description="Maximum consensus checkpoints to backtrack through upon sub-threshold simulation.",
    )
    telemetry_enabled: bool = Field(
        default=True,
        description="Whether to emit WebSocket telemetry events during recovery.",
    )
    mediator_agent_id: str = Field(
        default="counterfactual_resolver",
        description="Agent ID assigned to the automated compromise author.",
    )
    mediator_role: AgentRole = Field(
        default=AgentRole.COUNTERFACTUAL,
        description="Domain role assigned to the automated compromise author.",
    )


class DeadlockSolverDaemon:
    """
    Autonomous recovery daemon bridging the scheduler's deadlock hook
    to the counterfactual sandbox simulation and credit assignment pipeline.
    """

    def __init__(
        self,
        config: Optional[SolverConfig] = None,
        broker: Optional[Any] = None,
        attribution_engine: Optional[DeadlockAttributionEngine] = None,
        sandbox_manager: Optional[SandboxManager] = None,
        simulator: Optional[CounterfactualSimulator] = None,
        scoring_engine: Optional[CreditAssignmentScorer] = None,
    ):
        self.config: SolverConfig = config or SolverConfig()
        self.broker: Any = broker
        self.attribution_engine: DeadlockAttributionEngine = attribution_engine or DeadlockAttributionEngine()
        self.sandbox_manager: SandboxManager = sandbox_manager or SandboxManager()
        self.simulator: CounterfactualSimulator = simulator or CounterfactualSimulator(
            target_score=self.config.convergence_threshold,
            max_iterations=self.config.max_simulation_rounds,
        )
        self.scoring_engine: CreditAssignmentScorer = scoring_engine or CreditAssignmentScorer(
            convergence_threshold=self.config.convergence_threshold
        )
        self._attached_scheduler: Optional[Any] = None
        self._recovery_history: List[Dict[str, Any]] = []

    # --------------------------------------------------------------------------
    # Scheduler Lifecycle Attachment
    # --------------------------------------------------------------------------

    def attach_to_scheduler(self, scheduler: Any) -> None:
        """
        Hooks the daemon into a DeterministicScheduler instance.
        Whenever the scheduler halts on deadlock, it will invoke this daemon.
        """
        self._attached_scheduler = scheduler
        if hasattr(scheduler, "register_deadlock_hook"):
            scheduler.register_deadlock_hook(self)
            logger.info("DeadlockSolverDaemon attached to scheduler session %s", getattr(scheduler, "session_id", "unknown"))

    def detach(self) -> None:
        """Detaches from the currently attached scheduler and cleans up sandbox resources."""
        if self._attached_scheduler and hasattr(self._attached_scheduler, "_deadlock_hooks"):
            if self in self._attached_scheduler._deadlock_hooks:
                self._attached_scheduler._deadlock_hooks.remove(self)
        self._attached_scheduler = None
        self.cleanup()

    def cleanup(self) -> None:
        """Frees all tracked sandbox sessions from memory."""
        if self.sandbox_manager:
            self.sandbox_manager.cleanup_all()

    # --------------------------------------------------------------------------
    # Callable Hook Interface
    # --------------------------------------------------------------------------

    def __call__(self, notification: DeadlockNotification) -> Optional[ReviseProposal]:
        """Callable hook matching scheduler's register_deadlock_hook interface."""
        return self.resolve(notification)

    # --------------------------------------------------------------------------
    # Core Recovery Execution Pipeline
    # --------------------------------------------------------------------------

    def resolve(self, notification: DeadlockNotification) -> Optional[ReviseProposal]:
        """
        Executes the complete counterfactual recovery pipeline:
        1. Extract entries from notification.snapshot.
        2. Run attribution engine to discover divergence node and consensus checkpoints.
        3. Fork sandbox session at divergence point.
        4. Run adaptive simulator with plateau early-stopping.
        5. If score < threshold, escalate to prior consensus checkpoints.
        6. Score the winning branch using CreditAssignmentScorer.
        7. Package winning proposal into ReviseProposal.
        8. Emit telemetry and return proposal for Priority 0 preemption.
        """
        snapshot = notification.snapshot
        session_id = notification.session_id

        # 1. Convert snapshot contributions to BlackboardEntry list
        raw_contribs = snapshot.get_branch_entries("main") if hasattr(snapshot, "get_branch_entries") else snapshot.contributions
        entries: List[BlackboardEntry] = []
        for c in raw_contribs:
            if isinstance(c, BlackboardEntry):
                entries.append(copy.deepcopy(c))
            elif hasattr(c, "to_agent_contribution"):
                entries.append(BlackboardEntry.from_agent_contribution(c.to_agent_contribution(session_id=session_id)))
            elif isinstance(c, AgentContribution):
                entries.append(BlackboardEntry.from_agent_contribution(c))
            elif isinstance(c, dict):
                entries.append(BlackboardEntry.model_validate(c))

        if not entries:
            logger.warning("Deadlock notification contained no contributions. Recovery aborted.")
            return None

        # 2. Run Attribution Engine
        attribution: AttributionResult = self.attribution_engine.analyze(entries)
        if not attribution.deadlock_detected or not attribution.divergence_entry_id:
            logger.info("Attribution engine detected no structural divergence node for session %s.", session_id)
            return None

        # 3. Create InMemoryBlackboard representation of live history for sandbox manager
        live_board = InMemoryBlackboard(
            session_id=session_id,
            task_id=getattr(snapshot, "task_id", session_id),
            problem_statement=getattr(snapshot, "task_description", f"Task {session_id}"),
            ground_truth=snapshot.metadata.get("ground_truth") if snapshot.metadata else None,
        )
        for e in entries:
            live_board.add_entry(e)

        # 4. Initialize Sandbox Session
        sandbox_session = self.sandbox_manager.create_session_from_attribution(
            live_board=live_board,
            attribution=attribution,
        )

        # Extract deadlocked conflict window
        deadlock_window_size = len(notification.repeated_tags) if notification.repeated_tags else 2
        deadlock_entries = entries[-deadlock_window_size:]

        # 5. Multi-Tier Escalation Simulation Loop
        winning_candidate = None
        winning_score: Optional[BranchScoreDetails] = None
        winning_session: Optional[SandboxSession] = None
        escalation_level = 0

        broker = self.broker
        if broker is None:
            # Fallback to internal test mock broker if none provided
            from fixtures.mocks import MockLLMBroker
            broker = MockLLMBroker()

        while escalation_level < self.config.max_escalation_levels:
            # Run simulation on this sandbox tier
            sim_result = self.simulator.run_adaptive_search(
                sandbox_session=sandbox_session,
                attribution=attribution,
                broker=broker,
            )

            sim_entries = sandbox_session.get_simulated_entries()

            # Score simulated branch using Task 4 CreditAssignmentScorer
            score_details = self.scoring_engine.score_branch(
                deadlock_entries=deadlock_entries,
                simulated_entries=sim_entries,
                problem_context=getattr(snapshot, "task_description", None),
                context_evidence=attribution.clashing_evidence,
            )

            logger.info(
                "Sandbox branch %s (escalation %d) scored %f (status: %s)",
                sandbox_session.sandbox_branch_id,
                sandbox_session.current_checkpoint_level,
                score_details.composite_score,
                score_details.outcome_status,
            )

            # Check if convergence threshold met
            if score_details.composite_score >= self.config.convergence_threshold and sim_result.winning_candidate:
                winning_candidate = sim_result.winning_candidate
                winning_score = score_details
                winning_session = sandbox_session
                break

            # Try escalating to prior consensus checkpoint if available
            can_escalate = sandbox_session.escalate_to_prior_checkpoint()
            if not can_escalate:
                break
            escalation_level += 1

        # 6. Verify winning resolution
        if not winning_candidate or not winning_score or not winning_session:
            logger.warning(
                "Counterfactual resolution exhausted %d escalation levels without reaching threshold %f.",
                escalation_level,
                self.config.convergence_threshold,
            )
            return None

        # 7. Formulate Winning ReviseProposal
        rev_entry = winning_candidate.revised_entry

        # Format revised AgentContribution authored by mediator
        revised_contrib = AgentContribution(
            contribution_id=f"c_rev_{uuid.uuid4().hex[:8]}",
            session_id=session_id,
            turn_index=notification.deadlock_turn_index + 1,
            agent_id=self.config.mediator_agent_id,
            agent_role=self.config.mediator_role,
            tag=PXPTag.REVISE,
            target_contribution_id=attribution.divergence_entry_id,
            payload=PEXPayload(
                prediction=Prediction(
                    claim=rev_entry.prediction,
                    confidence=rev_entry.confidence,
                ),
                explanation=Explanation(
                    rationale=rev_entry.explanation,
                    evidence=rev_entry.evidence_refs,
                ),
            ),
            is_counterfactual=True,
            metadata={
                "winning_branch_id": winning_session.sandbox_branch_id,
                "divergence_entry_id": attribution.divergence_entry_id,
                "composite_score": winning_score.composite_score,
                "escalation_level": winning_session.current_checkpoint_level,
            },
        )

        # Attribute causality credit/blame
        credit_deltas: List[CreditAssignmentDelta] = self.scoring_engine.assign_credit(
            deadlock_entries=deadlock_entries,
            simulated_entries=winning_session.get_simulated_entries(),
            composite_score=winning_score.composite_score,
        )

        proposal = ReviseProposal(
            session_id=session_id,
            winning_branch_id=winning_session.sandbox_branch_id,
            target_turn_index=attribution.rollback_step,
            revised_contribution=revised_contrib,
            credit_assignment_summary=(
                f"Counterfactual simulation on branch {winning_session.sandbox_branch_id} "
                f"at escalation level {winning_session.current_checkpoint_level} achieved convergence score "
                f"{winning_score.composite_score:.2f} (Agreement: {winning_score.agreement_ratio*100:.0f}%, "
                f"Entropy Drop: {winning_score.delta_tag_entropy:.2f}). "
                f"Attributed {len(credit_deltas)} causal credit/blame deltas."
            ),
        )

        # 8. Emit Telemetry
        if self.config.telemetry_enabled and self._attached_scheduler and hasattr(self._attached_scheduler, "_emit_telemetry"):
            self._attached_scheduler._emit_telemetry(
                event_type=TelemetryEventType.COUNTERFACTUAL_RESOLVED,
                payload={
                    "session_id": session_id,
                    "winning_branch_id": winning_session.sandbox_branch_id,
                    "convergence_score": winning_score.composite_score,
                    "outcome_status": winning_score.outcome_status,
                    "mediator_agent_id": self.config.mediator_agent_id,
                    "divergence_entry_id": attribution.divergence_entry_id,
                    "escalation_level": winning_session.current_checkpoint_level,
                },
            )

        # 9. Record in history
        self._recovery_history.append({
            "session_id": session_id,
            "deadlock_turn_index": notification.deadlock_turn_index,
            "winning_branch_id": winning_session.sandbox_branch_id,
            "convergence_score": winning_score.composite_score,
            "escalation_level": winning_session.current_checkpoint_level,
            "proposal": proposal,
            "score_details": winning_score,
            "credit_deltas": credit_deltas,
        })

        logger.info(
            "DeadlockSolverDaemon generated ReviseProposal on branch %s (score: %.3f)",
            winning_session.sandbox_branch_id,
            winning_score.composite_score,
        )
        return proposal

    def get_recovery_history(self) -> List[Dict[str, Any]]:
        """Returns the full history of deadlock resolutions performed by this daemon."""
        return [copy.deepcopy(item) for item in self._recovery_history]
