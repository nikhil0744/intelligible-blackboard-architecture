"""
counterfactual/scoring.py
Task 4: Credit Assignment & Branch Scoring Engine.

Implements information-theoretic entropy evaluation, context fidelity / semantic drift
penalties, trajectory plateau detection, and per-agent causal credit/blame attribution
aligned with Donald Michie's and Baskar et al.'s intelligibility framework.
"""

from __future__ import annotations

from collections import Counter
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field

from contracts.schemas import (
    AgentContribution,
    AgentRole,
    BlackboardEntry,
    BlackboardStatus,
    CounterfactualBranch,
    CreditAssignmentDelta,
    Explanation,
    PEXPayload,
    Prediction,
    PXPTag,
)


class BranchScoreDetails(BaseModel):
    """
    Detailed analytical breakdown of a simulated counterfactual branch evaluation.
    """
    model_config = ConfigDict(extra="ignore")

    agreement_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of simulated turns resulting in formal RATIFY consensus.",
    )
    tag_entropy_deadlock: float = Field(
        ...,
        ge=0.0,
        description="Shannon entropy (bits) of tag distribution in the deadlocked window.",
    )
    tag_entropy_branch: float = Field(
        ...,
        ge=0.0,
        description="Shannon entropy (bits) of tag distribution across simulated branch turns.",
    )
    delta_tag_entropy: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized reduction in communicative disorder (H_deadlock - H_branch) / H_max.",
    )
    mean_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Mean calibrated confidence asserted across simulated contributions.",
    )
    drift_penalty: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Semantic drift penalty quantifying deviation from case facts and clinical evidence.",
    )
    composite_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Final weighted composite evaluation score in range [0.0, 1.0].",
    )
    outcome_status: str = Field(
        ...,
        description="Outcome classification: 'CONVERGED', 'PLATEAUED', or 'FAILED'.",
    )
    weights_applied: Dict[str, float] = Field(
        default_factory=dict,
        description="Weights applied across the 4 objective pillars during evaluation.",
    )


class PlateauDetector:
    """
    Second-order trajectory analyzer tracking score velocity, acceleration,
    and rolling variance to trigger early stopping on asymptotic plateaus.

    Note:
        Instances maintain mutable internal trajectory state (`history`, `_stagnant_count`).
        Instantiate a fresh PlateauDetector per independent search branch.
    """

    def __init__(
        self,
        patience: int = 2,
        epsilon: float = 0.05,
        window_size: int = 3,
        min_velocity: float = 0.02,
    ):
        self.patience: int = max(1, patience)
        self.epsilon: float = max(0.0001, epsilon)
        self.window_size: int = max(2, window_size)
        self.min_velocity: float = min_velocity
        self.history: List[float] = []
        self._stagnant_count: int = 0

    def record_score(self, score: float) -> Tuple[bool, str]:
        """
        Records an evaluation score and determines if search improvement has plateaued.

        Returns:
            Tuple of (is_plateau, explanation_message).
        """
        self.history.append(score)
        n = len(self.history)

        if n < self.window_size:
            return False, f"Accumulating trajectory history ({n}/{self.window_size} rounds)."

        # First-order velocity
        velocity = self.history[-1] - self.history[-2]

        # Second-order acceleration (if at least 3 points)
        acceleration = 0.0
        if n >= 3:
            prev_velocity = self.history[-2] - self.history[-3]
            acceleration = velocity - prev_velocity

        # Rolling variance over window
        window = self.history[-self.window_size:]
        mean_score = sum(window) / len(window)
        variance = sum((s - mean_score) ** 2 for s in window) / len(window)

        # Plateau criteria: negligible variance and (velocity below threshold or decelerating)
        is_stagnant = (variance <= self.epsilon) and (velocity < self.min_velocity or acceleration <= 0.0)
        if is_stagnant:
            self._stagnant_count += 1
        else:
            self._stagnant_count = max(0, self._stagnant_count - 1)

        if self._stagnant_count >= self.patience:
            explanation = (
                f"Plateau detected at round {n}: rolling variance ({variance:.6f}) <= {self.epsilon} "
                f"with velocity {velocity:.4f} and acceleration {acceleration:.4f} "
                f"across {self._stagnant_count} consecutive rounds."
            )
            return True, explanation

        return False, f"Search progressing (velocity={velocity:.4f}, acceleration={acceleration:.4f}, variance={variance:.6f})."

    def reset(self) -> None:
        """Resets the trajectory history and counter."""
        self.history.clear()
        self._stagnant_count = 0


class CreditAssignmentScorer:
    """
    Evaluates simulated counterfactual branches and attributes causal credit/blame
    across participating agents based on information entropy and evidence retention.
    """

    MAX_ENTROPY: float = 2.0  # log2(4) for the 4 PXP tags (RATIFY, REVISE, REFUTE, REJECT)

    def __init__(
        self,
        w_agreement: float = 0.50,
        w_entropy: float = 0.25,
        w_confidence: float = 0.15,
        w_drift: float = 0.10,
        convergence_threshold: float = 0.70,
    ):
        self.w_agreement: float = w_agreement
        self.w_entropy: float = w_entropy
        self.w_confidence: float = w_confidence
        self.w_drift: float = w_drift
        self.convergence_threshold: float = convergence_threshold

    # --------------------------------------------------------------------------
    # Information-Theoretic Entropy
    # --------------------------------------------------------------------------

    @staticmethod
    def compute_shannon_entropy(tags: Iterable[Any]) -> float:
        """
        Computes Claude Shannon's information entropy in bits for a sequence of PXP tags.
        H(X) = -sum(P(x) * log2(P(x)))
        """
        tag_list = [str(t.value if hasattr(t, "value") else t).upper() for t in tags if t]
        if not tag_list:
            return 0.0

        n = len(tag_list)
        counts = Counter(tag_list)
        entropy = 0.0
        for count in counts.values():
            p = count / n
            if p > 0.0:
                entropy -= p * math.log2(p)

        return round(entropy, 4)

    # --------------------------------------------------------------------------
    # Semantic Drift & Context Retention
    # --------------------------------------------------------------------------

    @staticmethod
    def _extract_tokens(text: Optional[str]) -> Set[str]:
        """Extracts substantive lowercase words (length >= 3) ignoring formatting."""
        if not text:
            return set()
        return set(re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", text.lower()))

    def compute_drift_penalty(
        self,
        context_text: Optional[str],
        context_evidence: Optional[List[str]],
        branch_entries: List[Any],
    ) -> float:
        """
        Computes semantic drift penalty in [0.0, 1.0].
        A higher penalty indicates the simulated branch drifted away from the case facts.
        """
        if not branch_entries:
            return 1.0

        all_branch_text: List[str] = []
        branch_citations: Set[str] = set()

        for e in branch_entries:
            pred = getattr(e, "prediction", "") or ""
            expl = getattr(e, "explanation", "") or ""
            all_branch_text.append(f"{pred} {expl}")

            # Collect evidence citations if present (supports evidence_refs or evidence)
            ev_list = getattr(e, "evidence_refs", None) or getattr(e, "evidence", [])
            if isinstance(ev_list, list):
                for item in ev_list:
                    if isinstance(item, str):
                        clean_item = item.lower().strip()
                        branch_citations.add(clean_item)
                        branch_citations.add(clean_item.replace("_", " "))

        combined_text = " ".join(all_branch_text)
        branch_tokens = self._extract_tokens(combined_text)

        # 1. Evidence Retention
        valid_context_ev = [
            e.lower().strip()
            for e in (context_evidence or [])
            if e and isinstance(e, str) and e.strip()
        ]

        if valid_context_ev:
            retained = 0
            for ev_clean in valid_context_ev:
                ev_spaced = ev_clean.replace("_", " ")
                # Check direct citation or word-boundary regex in combined text
                if ev_clean in branch_citations or ev_spaced in branch_citations:
                    retained += 1
                elif (
                    re.search(rf"\b{re.escape(ev_clean)}\b", combined_text.lower())
                    or re.search(rf"\b{re.escape(ev_spaced)}\b", combined_text.lower())
                ):
                    retained += 1
            evidence_retention = retained / len(valid_context_ev)
        else:
            evidence_retention = 1.0

        # 2. Lexical Jaccard Overlap with problem context
        context_tokens = self._extract_tokens(context_text)
        if context_tokens and branch_tokens:
            intersection = len(context_tokens & branch_tokens)
            union = len(context_tokens | branch_tokens)
            lexical_overlap = intersection / union if union > 0 else 0.0
        else:
            lexical_overlap = 0.5  # Neutral default when no prompt text provided

        fidelity = 0.70 * evidence_retention + 0.30 * lexical_overlap
        drift = max(0.0, min(1.0, 1.0 - fidelity))
        return round(drift, 4)

    # --------------------------------------------------------------------------
    # Master Objective Scoring
    # --------------------------------------------------------------------------

    def score_branch(
        self,
        deadlock_entries: List[Any],
        simulated_entries: List[Any],
        problem_context: Optional[str] = None,
        context_evidence: Optional[List[str]] = None,
    ) -> BranchScoreDetails:
        """
        Evaluates a counterfactual branch across all 4 analytical pillars.
        Weights form a convex combination summing to 1.00:
        Score = w_agreement * AR + w_entropy * ΔH + w_confidence * MeanConf + w_drift * (1.0 - Drift)
        """
        if not simulated_entries:
            return BranchScoreDetails(
                agreement_ratio=0.0,
                tag_entropy_deadlock=0.0,
                tag_entropy_branch=0.0,
                delta_tag_entropy=0.0,
                mean_confidence=0.0,
                drift_penalty=1.0,
                composite_score=0.0,
                outcome_status="FAILED",
                weights_applied={
                    "w_agreement": self.w_agreement,
                    "w_entropy": self.w_entropy,
                    "w_confidence": self.w_confidence,
                    "w_drift": self.w_drift,
                },
            )

        # 1. Agreement Ratio & Mean Confidence
        ratify_count = 0
        total_conf = 0.0
        sim_tags = []

        for e in simulated_entries:
            tag_val = str(getattr(e, "tag", "")).upper()
            if "RATIFY" in tag_val:
                ratify_count += 1
            sim_tags.append(tag_val)
            raw_c = getattr(e, "confidence", None)
            conf = float(raw_c) if raw_c is not None else 0.5
            total_conf += conf

        num_sim = len(simulated_entries)
        agreement_ratio = round(ratify_count / num_sim, 4)
        mean_confidence = round(total_conf / num_sim, 4)

        # 2. Shannon Entropy & Delta
        dl_tags = [str(getattr(e, "tag", "")).upper() for e in deadlock_entries]
        h_deadlock = self.compute_shannon_entropy(dl_tags)
        h_branch = self.compute_shannon_entropy(sim_tags)

        # Delta entropy normalization
        if h_deadlock == 0.0 and h_branch == 0.0:
            delta_entropy = 0.50 if agreement_ratio >= 0.5 else 0.0
        else:
            delta_entropy = max(0.0, min(1.0, (h_deadlock - h_branch) / self.MAX_ENTROPY))
        delta_entropy = round(delta_entropy, 4)

        # 3. Semantic Drift Penalty
        drift_penalty = self.compute_drift_penalty(
            context_text=problem_context,
            context_evidence=context_evidence,
            branch_entries=simulated_entries,
        )
        fidelity = max(0.0, min(1.0, 1.0 - drift_penalty))

        # 4. Composite Objective Function (Convex Combination scaling to 1.00)
        composite = (
            self.w_agreement * agreement_ratio
            + self.w_entropy * delta_entropy
            + self.w_confidence * mean_confidence
            + self.w_drift * fidelity
        )
        composite_score = round(max(0.0, min(1.0, composite)), 4)

        # Outcome Status Classification
        if composite_score >= self.convergence_threshold and agreement_ratio >= 0.5:
            outcome_status = "CONVERGED"
        elif composite_score >= 0.35:
            outcome_status = "PLATEAUED"
        else:
            outcome_status = "FAILED"

        return BranchScoreDetails(
            agreement_ratio=agreement_ratio,
            tag_entropy_deadlock=h_deadlock,
            tag_entropy_branch=h_branch,
            delta_tag_entropy=delta_entropy,
            mean_confidence=mean_confidence,
            drift_penalty=drift_penalty,
            composite_score=composite_score,
            outcome_status=outcome_status,
            weights_applied={
                "w_agreement": self.w_agreement,
                "w_entropy": self.w_entropy,
                "w_confidence": self.w_confidence,
                "w_drift": self.w_drift,
            },
        )

    # --------------------------------------------------------------------------
    # Causality & Credit Assignment
    # --------------------------------------------------------------------------

    def assign_credit(
        self,
        deadlock_entries: List[Any],
        simulated_entries: Optional[List[Any]] = None,
        composite_score: float = 0.80,
    ) -> List[CreditAssignmentDelta]:
        """
        Attributes causal credit and blame across agents.
        - Blame in range [0.0, +1.0]: obstinate dispute instigators in the deadlock window.
        - Credit in range [-1.0, 0.0]: constructive synthesizers in the simulated branch.
        """
        deltas: List[CreditAssignmentDelta] = []

        # 1. Blame attribution in the deadlocked window
        for idx, entry in enumerate(deadlock_entries):
            agent_id = getattr(entry, "agent_id", "unknown_agent")
            cid = getattr(entry, "contribution_id", None) or getattr(entry, "entry_id", f"c_dl_{idx}")

            # Safe turn index extraction (handles turn_index = 0)
            raw_turn = getattr(entry, "turn_index", None)
            if raw_turn is None:
                raw_turn = getattr(entry, "step_number", None)
            turn_idx = int(raw_turn) if raw_turn is not None else idx

            tag_str = str(getattr(entry, "tag", "")).upper()

            # Safe confidence extraction (handles confidence = 0.0)
            raw_conf = getattr(entry, "confidence", None)
            conf = float(raw_conf) if raw_conf is not None else 0.5

            ev = getattr(entry, "evidence_refs", None) or getattr(entry, "evidence", []) or []

            # Base tag hostility
            if "REJECT" in tag_str:
                blame = 0.65
            elif "REFUTE" in tag_str:
                blame = 0.45
            elif "REVISE" in tag_str:
                blame = 0.15
            else:
                blame = 0.05

            # Evidence penalty: ungrounded rejection
            if len(ev) == 0 and ("REJECT" in tag_str or "REFUTE" in tag_str):
                blame += 0.20

            # Dogmatic overconfidence penalty
            if conf >= 0.85:
                blame += 0.15 * conf

            blame_score = round(min(1.0, blame), 3)
            rationale = (
                f"Agent {agent_id} exhibited {tag_str} stance with {conf*100:.0f}% confidence "
                f"and {len(ev)} evidence citations contributing to communicative deadlock."
            )

            deltas.append(
                CreditAssignmentDelta(
                    agent_id=agent_id,
                    turn_index=turn_idx,
                    contribution_id=cid,
                    causality_score=blame_score,
                    rationale=rationale,
                )
            )

        # 2. Resolution credit in the counterfactual simulated branch
        if simulated_entries:
            pivot_found = False
            for idx, entry in enumerate(simulated_entries):
                agent_id = getattr(entry, "agent_id", "counterfactual_synthesizer")
                cid = getattr(entry, "contribution_id", None) or getattr(entry, "entry_id", f"c_sim_{idx}")

                raw_turn = getattr(entry, "turn_index", None)
                if raw_turn is None:
                    raw_turn = getattr(entry, "step_number", None)
                turn_idx = int(raw_turn) if raw_turn is not None else (idx + len(deadlock_entries))

                tag_str = str(getattr(entry, "tag", "")).upper()

                if not pivot_found and ("REVISE" in tag_str or "RATIFY" in tag_str):
                    credit = -(0.50 + 0.50 * composite_score)
                    credit_score = round(max(-1.0, min(-0.1, credit)), 3)
                    rationale = (
                        f"Agent {agent_id} introduced pivotal {tag_str} synthesis that broke the "
                        f"deadlock impasse and paved the path to group consensus."
                    )
                    pivot_found = True
                elif "RATIFY" in tag_str:
                    credit_score = -0.30
                    rationale = f"Agent {agent_id} supported convergence with {tag_str} assertion."
                else:
                    # Non-converging assertion in simulated branch receives 0.0 (neutral, no credit)
                    credit_score = 0.0
                    rationale = f"Agent {agent_id} asserted {tag_str} stance without facilitating consensus."

                deltas.append(
                    CreditAssignmentDelta(
                        agent_id=agent_id,
                        turn_index=turn_idx,
                        contribution_id=cid,
                        causality_score=credit_score,
                        rationale=rationale,
                    )
                )

        return deltas

    # --------------------------------------------------------------------------
    # Contract Construction
    # --------------------------------------------------------------------------

    def build_counterfactual_branch_contract(
        self,
        branch_id: str,
        session_id: str,
        forked_at_turn_index: int,
        divergence_agent_id: str,
        counterfactual_assertion: AgentContribution | BlackboardEntry | Dict[str, Any],
        simulated_contributions: List[AgentContribution | BlackboardEntry | Dict[str, Any]],
        score_details: BranchScoreDetails,
    ) -> CounterfactualBranch:
        """
        Builds a canonical CounterfactualBranch contract instance matching Student 1's schema.
        Handles AgentContribution, BlackboardEntry, and raw dictionary inputs.
        """
        # Convert counterfactual_assertion to AgentContribution
        if isinstance(counterfactual_assertion, AgentContribution):
            cf_assertion = counterfactual_assertion
        elif hasattr(counterfactual_assertion, "to_agent_contribution"):
            cf_assertion = counterfactual_assertion.to_agent_contribution(session_id=session_id)
        elif isinstance(counterfactual_assertion, dict):
            cf_assertion = BlackboardEntry.model_validate(counterfactual_assertion).to_agent_contribution(session_id=session_id)
        else:
            cf_assertion = BlackboardEntry.model_validate(counterfactual_assertion).to_agent_contribution(session_id=session_id)

        # Convert simulated contributions (supports dict, BlackboardEntry, AgentContribution)
        contribs: List[AgentContribution] = []
        for c in simulated_contributions:
            if isinstance(c, AgentContribution):
                contribs.append(c)
            elif hasattr(c, "to_agent_contribution"):
                contribs.append(c.to_agent_contribution(session_id=session_id))
            elif isinstance(c, BlackboardEntry):
                contribs.append(c.to_agent_contribution(session_id=session_id))
            elif isinstance(c, dict):
                contribs.append(BlackboardEntry.model_validate(c).to_agent_contribution(session_id=session_id))

        status_map = {
            "CONVERGED": BlackboardStatus.CONSENSUS,
            "RESOLVED": BlackboardStatus.RESOLVED,
            "PLATEAUED": BlackboardStatus.DEADLOCK,
            "FAILED": BlackboardStatus.TERMINATED,
        }
        mapped_status = status_map.get(
            score_details.outcome_status.upper(),
            BlackboardStatus.CONSENSUS if score_details.composite_score >= self.convergence_threshold else BlackboardStatus.TERMINATED
        )

        return CounterfactualBranch(
            branch_id=branch_id,
            session_id=session_id,
            forked_at_turn_index=forked_at_turn_index,
            divergence_agent_id=divergence_agent_id,
            counterfactual_assertion=cf_assertion,
            simulated_trajectory=contribs,
            outcome_status=mapped_status,
            convergence_score=score_details.composite_score,
        )
