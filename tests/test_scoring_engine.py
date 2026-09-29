"""
tests/test_scoring_engine.py
Comprehensive unit and integration test suite for Task 4:
Credit Assignment & Branch Scoring Engine.
"""

from typing import List
import pytest

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
from counterfactual.scoring import (
    BranchScoreDetails,
    CreditAssignmentScorer,
    PlateauDetector,
)


def _make_entry(
    agent_id: str,
    tag: PXPTag,
    claim: str = "Test claim",
    rationale: str = "Test rationale",
    confidence: float = 0.85,
    evidence: List[str] = None,
    turn_index: int = 1,
) -> BlackboardEntry:
    """Helper to construct a valid BlackboardEntry for scoring tests."""
    return BlackboardEntry(
        entry_id=f"entry_{agent_id}_{turn_index}",
        agent_id=agent_id,
        tag=tag,
        prediction=claim,
        explanation=rationale,
        confidence=confidence,
        evidence_refs=evidence or [],
        step_number=turn_index,
    )


# ============================================================================
# 1. Shannon Entropy Tests
# ============================================================================

def test_shannon_entropy_calculation():
    scorer = CreditAssignmentScorer()

    # 1. Uniform distribution across all 4 tags -> 2.0 bits
    uniform_tags = [PXPTag.RATIFY, PXPTag.REVISE, PXPTag.REFUTE, PXPTag.REJECT]
    assert scorer.compute_shannon_entropy(uniform_tags) == pytest.approx(2.0, abs=0.001)

    # 2. Pure consensus (all identical) -> 0.0 bits
    consensus_tags = [PXPTag.RATIFY, PXPTag.RATIFY, PXPTag.RATIFY]
    assert scorer.compute_shannon_entropy(consensus_tags) == 0.0

    # 3. Two-state conflict (50/50 REFUTE/REJECT) -> 1.0 bit
    dispute_tags = [PXPTag.REFUTE, PXPTag.REJECT, PXPTag.REFUTE, PXPTag.REJECT]
    assert scorer.compute_shannon_entropy(dispute_tags) == pytest.approx(1.0, abs=0.001)

    # 4. Empty tag list -> 0.0 bits
    assert scorer.compute_shannon_entropy([]) == 0.0


def test_entropy_reduction_delta():
    scorer = CreditAssignmentScorer()

    # Deadlock with 50/50 REFUTE/REJECT (H = 1.0 bit)
    dl_entries = [
        _make_entry("agent_a", PXPTag.REFUTE, turn_index=1),
        _make_entry("agent_b", PXPTag.REJECT, turn_index=2),
    ]

    # Branch with unanimous RATIFY (H = 0.0 bits)
    res_entries = [
        _make_entry("agent_a", PXPTag.RATIFY, turn_index=3),
        _make_entry("agent_b", PXPTag.RATIFY, turn_index=4),
    ]

    details = scorer.score_branch(dl_entries, res_entries)
    assert details.tag_entropy_deadlock == pytest.approx(1.0, abs=0.01)
    assert details.tag_entropy_branch == 0.0
    # Normalized delta = (1.0 - 0.0) / 2.0 = 0.50
    assert details.delta_tag_entropy == pytest.approx(0.50, abs=0.01)


def test_uniform_deadlock_absorbing_transition():
    scorer = CreditAssignmentScorer()

    # Both agents repeatedly REJECT (H = 0.0 bit)
    dl_entries = [
        _make_entry("agent_a", PXPTag.REJECT, turn_index=1),
        _make_entry("agent_b", PXPTag.REJECT, turn_index=2),
    ]

    # Simulated branch reaches unanimous RATIFY (H = 0.0 bit)
    res_entries = [
        _make_entry("agent_a", PXPTag.RATIFY, turn_index=3),
        _make_entry("agent_b", PXPTag.RATIFY, turn_index=4),
    ]

    details = scorer.score_branch(dl_entries, res_entries)
    assert details.agreement_ratio == 1.0
    # Absorbing state bonus applies
    assert details.delta_tag_entropy == 0.50
    assert details.outcome_status == "CONVERGED"


# ============================================================================
# 2. Semantic Drift & Evidence Retention Tests
# ============================================================================

def test_drift_penalty_preserves_case_context():
    scorer = CreditAssignmentScorer()

    context_prompt = (
        "Patient presents with subpleural honeycombing on HRCT and positive ANA titers. "
        "Differentiate idiopathic pulmonary fibrosis from systemic sclerosis-associated ILD."
    )
    context_evidence = ["hrct_honeycombing", "ana_positive"]

    # 1. Grounded branch discussing the findings
    grounded_branch = [
        _make_entry(
            "agent_a",
            PXPTag.RATIFY,
            claim="Systemic sclerosis associated ILD",
            rationale="Subpleural honeycombing on HRCT combined with positive ANA titers supports connective tissue disease.",
            evidence=["hrct_honeycombing", "ana_positive"],
        )
    ]
    drift_grounded = scorer.compute_drift_penalty(context_prompt, context_evidence, grounded_branch)
    assert drift_grounded < 0.25

    # 2. Sycophantic/evasive branch abandoning medical context
    evasive_branch = [
        _make_entry(
            "agent_a",
            PXPTag.RATIFY,
            claim="General healthy lifestyle recommendation",
            rationale="The weather is nice today, let us recommend vitamins and rest.",
            evidence=[],
        )
    ]
    drift_evasive = scorer.compute_drift_penalty(context_prompt, context_evidence, evasive_branch)
    assert drift_evasive > 0.60
    assert drift_evasive > drift_grounded


def test_empty_evidence_fallback():
    scorer = CreditAssignmentScorer()
    branch = [_make_entry("agent_a", PXPTag.RATIFY, claim="Logic conclusion", rationale="Follows from premises")]

    # When context_evidence is None or empty, no zero division occurs
    drift = scorer.compute_drift_penalty(context_text=None, context_evidence=None, branch_entries=branch)
    assert 0.0 <= drift <= 1.0


# ============================================================================
# 3. Master Objective Function & Weights
# ============================================================================

def test_branch_objective_function_weighting():
    scorer = CreditAssignmentScorer(
        w_agreement=0.50,
        w_entropy=0.25,
        w_confidence=0.15,
        w_drift=0.10,
        convergence_threshold=0.70,
    )

    dl_entries = [
        _make_entry("a1", PXPTag.REFUTE, confidence=0.90, turn_index=1),
        _make_entry("a2", PXPTag.REJECT, confidence=0.85, turn_index=2),
    ]

    # Perfect simulated resolution
    sim_entries = [
        _make_entry("a1", PXPTag.RATIFY, confidence=0.90, turn_index=3),
        _make_entry("a2", PXPTag.RATIFY, confidence=0.95, turn_index=4),
    ]

    details = scorer.score_branch(
        deadlock_entries=dl_entries,
        simulated_entries=sim_entries,
        problem_context="Test problem context",
    )

    assert details.agreement_ratio == 1.0
    assert details.delta_tag_entropy == 0.50
    assert details.mean_confidence == pytest.approx(0.925, abs=0.01)
    assert details.composite_score >= 0.70
    assert details.outcome_status == "CONVERGED"


def test_empty_simulated_branch_handling():
    scorer = CreditAssignmentScorer()
    dl_entries = [_make_entry("a1", PXPTag.REFUTE, turn_index=1)]
    details = scorer.score_branch(deadlock_entries=dl_entries, simulated_entries=[])

    assert details.composite_score == 0.0
    assert details.outcome_status == "FAILED"


# ============================================================================
# 4. Causality & Blame Attribution Tests
# ============================================================================

def test_credit_assignment_blame_and_resolution():
    scorer = CreditAssignmentScorer()

    # Deadlock window: Agent A refutes with evidence, Agent B dogmatically rejects with NO evidence
    dl_entries = [
        _make_entry("agent_doc_a", PXPTag.REFUTE, confidence=0.80, evidence=["ct_scan"], turn_index=1),
        _make_entry("agent_doc_b", PXPTag.REJECT, confidence=0.98, evidence=[], turn_index=2),
    ]

    # Simulated branch: Facilitator synthesizes a compromise, both doctors ratify
    sim_entries = [
        _make_entry("facilitator", PXPTag.REVISE, confidence=0.90, claim="Unified diagnosis", turn_index=3),
        _make_entry("agent_doc_a", PXPTag.RATIFY, confidence=0.92, turn_index=4),
        _make_entry("agent_doc_b", PXPTag.RATIFY, confidence=0.95, turn_index=5),
    ]

    deltas = scorer.assign_credit(dl_entries, sim_entries, composite_score=0.85)
    assert len(deltas) == 5

    # 1. Agent Doc B (dogmatic REJECT, 0 evidence, 98% conf) should receive the highest blame
    doc_b_delta = next(d for d in deltas if d.agent_id == "agent_doc_b" and d.turn_index == 2)
    assert doc_b_delta.causality_score > 0.80
    assert "REJECT" in doc_b_delta.rationale

    # 2. Agent Doc A had evidence and REFUTE, lower blame than Doc B
    doc_a_delta = next(d for d in deltas if d.agent_id == "agent_doc_a" and d.turn_index == 1)
    assert 0.0 < doc_a_delta.causality_score < doc_b_delta.causality_score

    # 3. Facilitator should receive strong negative causality (resolution credit)
    fac_delta = next(d for d in deltas if d.agent_id == "facilitator")
    assert fac_delta.causality_score <= -0.80
    assert "pivotal" in fac_delta.rationale.lower()

    # Ensure all conform to Student 1's Pydantic model
    for d in deltas:
        assert isinstance(d, CreditAssignmentDelta)


# ============================================================================
# 5. Plateau Detector Tests
# ============================================================================

def test_plateau_detector_dynamics():
    detector = PlateauDetector(patience=2, epsilon=0.005, window_size=3)

    # Round 1 & 2: Building history
    is_p, _ = detector.record_score(0.40)
    assert is_p is False
    is_p, _ = detector.record_score(0.55)
    assert is_p is False

    # Round 3: Still growing (velocity > 0)
    is_p, _ = detector.record_score(0.70)
    assert is_p is False

    # Round 4: First flat point (window has 0.55, 0.70, 0.701)
    is_p, _ = detector.record_score(0.701)
    assert is_p is False

    # Round 5: Window has 0.70, 0.701, 0.7012 (variance is tiny, stagnant count = 1)
    is_p, _ = detector.record_score(0.7012)
    assert is_p is False

    # Round 6: Window has 0.701, 0.7012, 0.7014 (stagnant count = 2 -> reaches patience)
    is_p, msg = detector.record_score(0.7014)
    assert is_p is True
    assert "Plateau detected" in msg

    # Reset
    detector.reset()
    assert len(detector.history) == 0


# ============================================================================
# 6. CounterfactualBranch Contract Serialization Test
# ============================================================================

def test_counterfactual_branch_contract_generation():
    scorer = CreditAssignmentScorer()

    sim_entries = [
        _make_entry("agent_1", PXPTag.RATIFY, turn_index=3),
        _make_entry("agent_2", PXPTag.RATIFY, turn_index=4),
    ]

    cf_assertion = _make_entry("agent_1", PXPTag.REVISE, claim="Compromise Claim", turn_index=2)

    details = BranchScoreDetails(
        agreement_ratio=1.0,
        tag_entropy_deadlock=1.0,
        tag_entropy_branch=0.0,
        delta_tag_entropy=0.50,
        mean_confidence=0.92,
        drift_penalty=0.05,
        composite_score=0.88,
        outcome_status="CONVERGED",
    )

    branch = scorer.build_counterfactual_branch_contract(
        branch_id="test_branch_001",
        session_id="test_session_123",
        forked_at_turn_index=2,
        divergence_agent_id="agent_1",
        counterfactual_assertion=cf_assertion,
        simulated_contributions=sim_entries,
        score_details=details,
    )

    assert isinstance(branch, CounterfactualBranch)
    assert branch.branch_id == "test_branch_001"
    assert branch.session_id == "test_session_123"
    assert branch.forked_at_turn_index == 2
    assert branch.divergence_agent_id == "agent_1"
    assert branch.convergence_score == 0.88
    assert branch.outcome_status == BlackboardStatus.CONSENSUS
    assert len(branch.simulated_trajectory) == 2
    assert isinstance(branch.simulated_trajectory[0], AgentContribution)
    assert isinstance(branch.counterfactual_assertion, AgentContribution)
