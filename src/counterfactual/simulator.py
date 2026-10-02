"""
counterfactual/simulator.py
Adaptive Simulated Rollouts and PXP Agreement Heuristic Engine.
Implements hill-climbing search with patience-based plateau detection,
evidence Jaccard scoring, and isolated sandbox rollout exploration.
"""

from enum import Enum
import json
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from contracts.schemas import AgentContribution, BlackboardEntry, PXPTag
from sandbox.manager import SandboxSession
from counterfactual.attribution import AttributionResult

try:
    from fixtures.mocks import BaseLLMBroker, MockLLMBroker
except ImportError:
    BaseLLMBroker = Any  # type: ignore

    class MockLLMBroker:  # type: ignore
        def __init__(self, default_response: Optional[Any] = None):
            self.default_response = default_response or {
                "tag": "RATIFY",
                "prediction": "Mock consensus prediction",
                "explanation": "Mock consensus explanation.",
                "evidence_refs": [],
                "confidence": 0.90,
            }

        def generate(self, prompt: Any) -> Any:
            return self.default_response


class SimulationStatus(str, Enum):
    """Outcome status of a counterfactual simulation search."""
    CONSENSUS_FOUND = "CONSENSUS_FOUND"
    PLATEAU_DETECTED = "PLATEAU_DETECTED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"


class SimulatedCandidate(BaseModel):
    """Pair of simulated revision and peer reaction generated in a search iteration."""
    iteration: int = Field(description="Search loop iteration index (1-based)")
    revised_entry: BlackboardEntry = Field(description="The synthesized revision proposed by the target agent")
    peer_reaction: BlackboardEntry = Field(description="The 1-hop simulated reaction from the opposing agent")
    score: float = Field(description="Agreement score S_k computed for this candidate pair")
    score_delta: float = Field(default=0.0, description="Score delta S_k - S_{k-1}")
    critique: Optional[str] = Field(default=None, description="Critique extracted from peer reaction if sub-target")


class SimulationResult(BaseModel):
    """Comprehensive diagnostic result returned by CounterfactualSimulator."""
    status: SimulationStatus = Field(description="Outcome status of the simulation")
    total_iterations: int = Field(description="Total number of simulated iterations executed")
    winning_candidate: Optional[SimulatedCandidate] = Field(
        default=None, description="Best candidate achieving consensus, or top-scoring candidate"
    )
    best_score: float = Field(default=0.0, description="Highest agreement score achieved during simulation")
    score_trajectory: List[float] = Field(
        default_factory=list, description="Sequence of agreement scores across iterations"
    )
    plateau_proven: bool = Field(
        default=False, 
        description="True if search terminated due to consecutive non-improving iterations >= patience"
    )
    reason: str = Field(description="Human/telemetry explanation of search outcome")


class CounterfactualSimulator:
    """
    Adaptive counterfactual simulation engine.
    Executes hill-climbing search over candidate revisions in an isolated sandbox,
    scoring multi-agent consensus via the PXP Agreement Heuristic Engine.
    """

    def __init__(
        self,
        target_score: float = 0.85,
        patience: int = 2,
        epsilon: float = 0.05,
        max_iterations: int = 6,
    ):
        self.target_score = target_score
        self.patience = patience
        self.epsilon = epsilon
        self.max_iterations = max_iterations

    def compute_agreement_score(
        self,
        revised_entry: BlackboardEntry,
        peer_reaction: BlackboardEntry,
        ground_truth: Optional[str] = None,
    ) -> float:
        """
        Computes composite PXP agreement score S_k in [0.0, 1.0].
        - Tag weight: RATIFY = 1.0, REVISE = 0.6, REFUTE = 0.2, REJECT = 0.0
        - Evidence Jaccard similarity: |E1 ∩ E2| / |E1 ∪ E2| (0.5 if both empty, 1.0 if identical)
        - Explanation depth & quality score
        - Overconfidence penalty if confidence > 0.9 without mutual evidence
        """
        # 1. Tag weight
        tag = peer_reaction.tag
        tag_val = tag.value if hasattr(tag, "value") else str(tag)
        tag_weights = {
            PXPTag.RATIFY.value: 1.0,
            PXPTag.REVISE.value: 0.6,
            PXPTag.REFUTE.value: 0.2,
            PXPTag.REJECT.value: 0.0,
        }
        tag_weight = tag_weights.get(tag_val, 0.5)

        # 2. Evidence Jaccard similarity
        e1 = set(revised_entry.evidence_refs or [])
        e2 = set(peer_reaction.evidence_refs or [])
        if not e1 and not e2:
            evidence_jaccard = 0.5
        elif e1 == e2:
            evidence_jaccard = 1.0
        else:
            intersection = len(e1 & e2)
            union = len(e1 | e2)
            evidence_jaccard = intersection / union if union > 0 else 0.5

        # 3. Explanation depth & quality
        expl_words = peer_reaction.explanation.split()
        depth_score = min(len(expl_words) / 20.0, 1.0)

        reasoning_markers = {
            "because", "therefore", "indicates", "suggests", "consistent",
            "evidence", "due to", "confirms", "shows", "demonstrates",
            "based on", "rules out", "secondary to", "pathognomonic"
        }
        lower_expl = peer_reaction.explanation.lower()
        marker_count = sum(1 for m in reasoning_markers if m in lower_expl)
        marker_bonus = min(marker_count * 0.05, 0.20)
        explanation_quality = min(depth_score * 0.80 + marker_bonus, 1.0)

        if ground_truth and ground_truth.strip():
            gt_lower = ground_truth.strip().lower()
            if (
                gt_lower in revised_entry.prediction.lower()
                or gt_lower in peer_reaction.prediction.lower()
                or gt_lower in lower_expl
            ):
                explanation_quality = min(explanation_quality + 0.10, 1.0)

        # 4. Overconfidence penalty
        has_mutual_evidence = len(e1 & e2) > 0
        overconfidence_penalty = 0.0
        if (peer_reaction.confidence > 0.9 or revised_entry.confidence > 0.9) and not has_mutual_evidence:
            overconfidence_penalty = 0.15

        # 5. Composite normalization
        w_tag = 0.50
        w_evid = 0.25
        w_expl = 0.25

        raw_score = (
            (tag_weight * w_tag)
            + (evidence_jaccard * w_evid)
            + (explanation_quality * w_expl)
            - overconfidence_penalty
        )
        normalized_score = max(0.0, min(1.0, raw_score))
        return round(float(normalized_score), 4)

    def generate_synthesis_prompt(
        self,
        attribution: AttributionResult,
        target_agent_id: str,
        prior_critique: Optional[str] = None,
    ) -> str:
        """
        Formats a retrospective hindsight prompt exposing the conflict,
        contested prediction, peer prediction, clashing evidence, and prior critique.
        """
        target_pred = attribution.contested_predictions.get(target_agent_id, "None specified")
        peer_preds = [
            f"{agent}: {pred}"
            for agent, pred in attribution.contested_predictions.items()
            if agent != target_agent_id
        ]
        peer_preds_str = "; ".join(peer_preds) if peer_preds else "None"
        clashing_ev_str = ", ".join(attribution.clashing_evidence) if attribution.clashing_evidence else "None"

        critique_section = ""
        if prior_critique and prior_critique.strip():
            critique_section = (
                f"\n[PRIOR PEER CRITIQUE]\n"
                f"Address the following critique from your peer's prior reaction:\n"
                f"\"{prior_critique.strip()}\"\n"
            )

        prompt = (
            f"[RETROSPECTIVE SYNTHESIS PROMPT]\n"
            f"Target Agent: {target_agent_id}\n"
            f"Context: An unresolvable debate impasse was detected.\n"
            f"Reason: {attribution.reason}\n\n"
            f"[CONFLICT DETAILS]\n"
            f"- Your Contested Position: {target_pred}\n"
            f"- Opposing Peer Positions: {peer_preds_str}\n"
            f"- Clashing Evidence Cited: {clashing_ev_str}\n"
            f"{critique_section}\n"
            f"[TASK]\n"
            f"Formulate a synthesized revision (PXPTag.REVISE) that resolves the contradiction. "
            f"Integrate the clashing evidence into a coherent causal explanation. "
            f"Provide your updated prediction, revised explanation, cited evidence references, and calibrated confidence."
        )
        return prompt

    def run_adaptive_search(
        self,
        sandbox_session: SandboxSession,
        attribution: AttributionResult,
        broker: Optional[Any] = None,
        target_agent_id: Optional[str] = None,
        opposing_agent_id: Optional[str] = None,
    ) -> SimulationResult:
        """
        Executes hill-climbing search in the isolated sandbox session up to max_iterations.
        Iteratively generates synthesized revisions and 1-hop peer reactions,
        computing agreement scores and terminating on consensus, plateau, or budget exhaustion.
        """
        if broker is None:
            broker = MockLLMBroker()

        # Resolve conflicting agents without self-collision
        if target_agent_id is None:
            target_agent_id = attribution.conflicting_agents[0] if attribution.conflicting_agents else "agent_target"

        if opposing_agent_id is None:
            potential_opponents = [a for a in attribution.conflicting_agents if a != target_agent_id]
            if potential_opponents:
                opposing_agent_id = potential_opponents[0]
            elif len(attribution.conflicting_agents) > 1:
                opposing_agent_id = attribution.conflicting_agents[1]
            else:
                opposing_agent_id = "agent_opposing"

        # Resolve roles from existing sandbox history
        target_role = self._resolve_agent_role(sandbox_session, target_agent_id, default="generalist")
        opposing_role = self._resolve_agent_role(sandbox_session, opposing_agent_id, default="generalist")

        score_trajectory: List[float] = []
        best_score: float = 0.0
        best_candidate: Optional[SimulatedCandidate] = None
        prior_critique: Optional[str] = None
        consecutive_non_improving: int = 0

        board_state = (
            sandbox_session.board.get_state()
            if hasattr(sandbox_session.board, "get_state")
            else getattr(sandbox_session.board, "_state", None)
        )
        ground_truth = getattr(board_state, "ground_truth", None) if board_state else None

        for iteration in range(1, self.max_iterations + 1):
            # 1. Generate synthesis prompt for target agent
            synthesis_prompt = self.generate_synthesis_prompt(
                attribution=attribution,
                target_agent_id=target_agent_id,
                prior_critique=prior_critique,
            )

            # 2. Query broker for target agent's revised entry
            rev_raw = self._query_broker(broker, synthesis_prompt, target_agent_id)
            rev_fields = self._extract_entry_fields(rev_raw)
            clean_rev = self._sanitize_entry_kwargs(
                fields=rev_fields,
                fallback_prediction=f"Synthesized revision round {iteration}",
                fallback_explanation=f"Synthesis addressing clashing evidence at round {iteration}.",
                fallback_evidence=list(attribution.clashing_evidence),
                default_confidence=0.85,
            )

            existing_entries = sandbox_session.get_all_entries()
            last_entry_id = (
                existing_entries[-1].entry_id if existing_entries else attribution.divergence_entry_id
            )
            max_step = max((e.step_number for e in existing_entries), default=sandbox_session.rollback_step)
            rev_step = max_step + 1

            revised_entry = BlackboardEntry(
                entry_id=f"sim_rev_{uuid.uuid4().hex[:8]}",
                parent_id=last_entry_id,
                branch_id=sandbox_session.sandbox_branch_id,
                agent_id=target_agent_id,
                agent_role=rev_fields.get("agent_role") or target_role,
                tag=PXPTag.REVISE,
                prediction=clean_rev["prediction"],
                explanation=clean_rev["explanation"],
                evidence_refs=clean_rev["evidence_refs"],
                confidence=clean_rev["confidence"],
                step_number=rev_step,
                metadata={
                    "is_simulated": True,
                    "iteration": iteration,
                    **clean_rev["metadata"],
                },
            )
            added_revised = sandbox_session.add_simulated_entry(revised_entry)

            # 3. Query broker for opposing agent's 1-hop reaction
            peer_prompt = (
                f"[REACTION PROMPT]\n"
                f"Agent: {opposing_agent_id} ({opposing_role})\n"
                f"Review the revised proposal from {target_agent_id}:\n"
                f"- Prediction: {added_revised.prediction}\n"
                f"- Explanation: {added_revised.explanation}\n"
                f"- Cited Evidence: {', '.join(added_revised.evidence_refs)}\n"
                f"Provide your 1-hop PXP reaction (RATIFY, REVISE, REFUTE, or REJECT)."
            )
            peer_raw = self._query_broker(broker, peer_prompt, opposing_agent_id)
            peer_fields = self._extract_entry_fields(peer_raw)
            clean_peer = self._sanitize_entry_kwargs(
                fields=peer_fields,
                fallback_prediction=added_revised.prediction,
                fallback_explanation="Peer evaluated synthesized revision.",
                fallback_evidence=added_revised.evidence_refs,
                default_confidence=0.90,
            )

            peer_tag_val = peer_fields.get("tag", PXPTag.RATIFY.value)
            try:
                peer_tag = PXPTag(peer_tag_val)
            except (ValueError, KeyError):
                peer_tag = PXPTag.RATIFY

            peer_step = rev_step + 1
            peer_reaction = BlackboardEntry(
                entry_id=f"sim_peer_{uuid.uuid4().hex[:8]}",
                parent_id=added_revised.entry_id,
                branch_id=sandbox_session.sandbox_branch_id,
                agent_id=opposing_agent_id,
                agent_role=peer_fields.get("agent_role") or opposing_role,
                tag=peer_tag,
                prediction=clean_peer["prediction"],
                explanation=clean_peer["explanation"],
                evidence_refs=clean_peer["evidence_refs"],
                confidence=clean_peer["confidence"],
                step_number=peer_step,
                metadata={
                    "is_simulated": True,
                    "iteration": iteration,
                    **clean_peer["metadata"],
                },
            )
            added_peer = sandbox_session.add_simulated_entry(peer_reaction)

            # 4. Compute score S_k
            score = self.compute_agreement_score(added_revised, added_peer, ground_truth=ground_truth)

            # 5. Delta S_k = S_k - S_{k-1}
            if score_trajectory:
                score_delta = round(score - score_trajectory[-1], 4)
            else:
                score_delta = 0.0

            score_trajectory.append(score)

            # 6. Extract critique for next iteration if sub-target
            if score < self.target_score:
                critique = added_peer.explanation
                prior_critique = critique
            else:
                critique = None
                prior_critique = None

            candidate = SimulatedCandidate(
                iteration=iteration,
                revised_entry=added_revised,
                peer_reaction=added_peer,
                score=score,
                score_delta=score_delta,
                critique=critique,
            )

            if score > best_score or best_candidate is None:
                best_score = score
                best_candidate = candidate

            # 7. Check if consensus reached
            if score >= self.target_score:
                return SimulationResult(
                    status=SimulationStatus.CONSENSUS_FOUND,
                    total_iterations=iteration,
                    winning_candidate=candidate,
                    best_score=score,
                    score_trajectory=score_trajectory,
                    plateau_proven=False,
                    reason=(
                        f"Consensus achieved at iteration {iteration} with agreement score "
                        f"{score:.4f} >= target {self.target_score:.4f}."
                    ),
                )

            # 8. Track plateau (consecutive iterations where score_delta < epsilon)
            if iteration > 1:
                if score_delta < self.epsilon:
                    consecutive_non_improving += 1
                else:
                    consecutive_non_improving = 0

                if consecutive_non_improving >= self.patience:
                    return SimulationResult(
                        status=SimulationStatus.PLATEAU_DETECTED,
                        total_iterations=iteration,
                        winning_candidate=best_candidate,
                        best_score=best_score,
                        score_trajectory=score_trajectory,
                        plateau_proven=True,
                        reason=(
                            f"Plateau detected after {consecutive_non_improving} consecutive non-improving "
                            f"iterations (score delta < {self.epsilon}). Search terminated early at iteration {iteration}."
                        ),
                    )

        # 9. Budget exhausted
        return SimulationResult(
            status=SimulationStatus.BUDGET_EXHAUSTED,
            total_iterations=self.max_iterations,
            winning_candidate=best_candidate,
            best_score=best_score,
            score_trajectory=score_trajectory,
            plateau_proven=False,
            reason=(
                f"Search budget exhausted after {self.max_iterations} iterations without reaching "
                f"target score {self.target_score:.4f} (best score: {best_score:.4f})."
            ),
        )

    def simulate(
        self,
        sandbox_session: SandboxSession,
        attribution: AttributionResult,
        broker: Optional[Any] = None,
        target_agent_id: Optional[str] = None,
        opposing_agent_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SimulationResult:
        """Alias for run_adaptive_search accepting optional kwargs for backwards compatibility."""
        return self.run_adaptive_search(
            sandbox_session=sandbox_session,
            attribution=attribution,
            broker=broker,
            target_agent_id=target_agent_id,
            opposing_agent_id=opposing_agent_id,
        )

    def _resolve_agent_role(self, sandbox_session: SandboxSession, agent_id: str, default: str = "generalist") -> str:
        """Finds existing role for agent_id in sandbox history."""
        for entry in sandbox_session.get_all_entries():
            if entry.agent_id == agent_id and entry.agent_role:
                return entry.agent_role
        return default

    def _extract_entry_fields(self, res: Any) -> Dict[str, Any]:
        """Polymorphic helper extracting field dictionary from broker output."""
        if hasattr(res, "text") and isinstance(res.text, str):
            res = res.text

        if isinstance(res, AgentContribution):
            return {
                "tag": res.tag,
                "prediction": res.prediction,
                "explanation": res.explanation,
                "evidence_refs": res.evidence_refs,
                "confidence": res.confidence,
                "agent_role": res.metadata.get("persona") or (res.agent_role.value if res.agent_role else "generalist"),
                "metadata": res.metadata,
            }

        if isinstance(res, BlackboardEntry):
            return {
                "tag": res.tag,
                "prediction": res.prediction,
                "explanation": res.explanation,
                "evidence_refs": res.evidence_refs,
                "confidence": res.confidence,
                "agent_role": res.agent_role,
                "metadata": res.metadata,
            }
        if isinstance(res, str):
            try:
                from prompts.parser import extract_json
                return extract_json(res)
            except Exception:
                pass
            try:
                parsed = json.loads(res.strip())
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
        if isinstance(res, dict):
            return res
        return {
            "tag": getattr(res, "tag", PXPTag.RATIFY),
            "prediction": getattr(res, "prediction", "Synthesized prediction"),
            "explanation": getattr(res, "explanation", "Synthesized explanation."),
            "evidence_refs": getattr(res, "evidence_refs", []),
            "confidence": getattr(res, "confidence", 0.90),
            "agent_role": getattr(res, "agent_role", "generalist"),
            "metadata": getattr(res, "metadata", {}),
        }

    def _query_broker(self, broker: Any, prompt: str, agent_id: str) -> Any:
        """
        Polymorphic query helper supporting test brokers (str -> dict/str)
        and Student 2's production ModelBroker (LLMRequest -> LLMResponse).
        """
        if hasattr(broker, "generate"):
            try:
                return broker.generate(prompt)
            except (TypeError, AttributeError):
                pass
            try:
                from llm_broker.types import ChatMessage, LLMRequest
                req = LLMRequest(
                    messages=[ChatMessage(role="user", content=prompt)],
                    agent_id=agent_id,
                )
                resp = broker.generate(req)
                return getattr(resp, "text", resp)
            except Exception as e:
                raise RuntimeError(f"Error querying broker with LLMRequest: {e}") from e
        return broker(prompt)

    def _sanitize_entry_kwargs(
        self,
        fields: Dict[str, Any],
        fallback_prediction: str,
        fallback_explanation: str,
        fallback_evidence: List[str],
        default_confidence: float = 0.85,
    ) -> Dict[str, Any]:
        """Sanitizes candidate field values to prevent Pydantic ValidationError or TypeError."""
        pred = fields.get("prediction")
        if not isinstance(pred, str) or not pred.strip():
            pred = fallback_prediction

        expl = fields.get("explanation")
        if not isinstance(expl, str) or not expl.strip():
            expl = fallback_explanation

        try:
            conf_val = fields.get("confidence")
            conf = float(conf_val) if conf_val is not None else default_confidence
            conf = max(0.0, min(1.0, conf))
        except (ValueError, TypeError):
            conf = default_confidence

        ev_refs = fields.get("evidence_refs")
        if not isinstance(ev_refs, list):
            ev_refs = list(fallback_evidence)

        meta = fields.get("metadata")
        if not isinstance(meta, dict):
            meta = {}

        return {
            "prediction": pred.strip(),
            "explanation": expl.strip(),
            "confidence": conf,
            "evidence_refs": ev_refs,
            "metadata": meta,
        }
