"""
tests/test_adaptive_simulator.py
Unit and integration tests for CounterfactualSimulator, adaptive rollouts,
and PXP agreement heuristic engine.
"""

import json
import pytest
from typing import Dict, List

from contracts.schemas import BlackboardEntry, PXPTag
from blackboard.store import InMemoryBlackboard
from counterfactual.attribution import AttributionResult, DeadlockAttributionEngine
from counterfactual.simulator import (
    CounterfactualSimulator,
    SimulatedCandidate,
    SimulationResult,
    SimulationStatus,
)
from sandbox.manager import SandboxManager, SandboxSession
from fixtures.mocks import MockLLMBroker


@pytest.fixture
def clinical_deadlock_board() -> InMemoryBlackboard:
    """Creates a realistic clinical deadlock blackboard based on sample_deadlocks.json."""
    board = InMemoryBlackboard(
        session_id="test_sess_clinical",
        task_id="med_task_fibrosis_vs_chf",
        problem_statement="Patient with progressive dyspnea, clubbing, bibasilar crackles, and HRCT honeycombing.",
        ground_truth="Idiopathic Pulmonary Fibrosis (IPF)",
    )
    board.add_entry(BlackboardEntry(
        entry_id="e1",
        parent_id=None,
        branch_id="main",
        agent_id="agent_cardio",
        agent_role="Cardiologist",
        tag=PXPTag.PROPOSE,
        prediction="Congestive Heart Failure (CHF)",
        explanation="Dyspnea and crackles indicate heart failure.",
        evidence_refs=["dyspnea", "crackles"],
        confidence=0.85,
        step_number=1,
    ))
    board.add_entry(BlackboardEntry(
        entry_id="e2",
        parent_id="e1",
        branch_id="main",
        agent_id="agent_pulmo",
        agent_role="Pulmonologist",
        tag=PXPTag.REFUTE,
        prediction="Idiopathic Pulmonary Fibrosis (IPF)",
        explanation="Refuting CHF. Honeycombing and clubbing indicate UIP/IPF. Normal BNP rules out CHF.",
        evidence_refs=["honeycombing", "clubbing", "normal_bnp"],
        confidence=0.95,
        step_number=2,
    ))
    board.add_entry(BlackboardEntry(
        entry_id="e3",
        parent_id="e2",
        branch_id="main",
        agent_id="agent_cardio",
        agent_role="Cardiologist",
        tag=PXPTag.REJECT,
        prediction="Congestive Heart Failure (CHF)",
        explanation="Rejecting pulmo refutation. Reticular pattern could be chronic congestion.",
        evidence_refs=["crackles", "cough"],
        confidence=0.80,
        step_number=3,
    ))
    board.add_entry(BlackboardEntry(
        entry_id="e4",
        parent_id="e3",
        branch_id="main",
        agent_id="agent_pulmo",
        agent_role="Pulmonologist",
        tag=PXPTag.REJECT,
        prediction="Idiopathic Pulmonary Fibrosis (IPF)",
        explanation="Rejecting cardio assertion. Subpleural honeycombing is structural fibrotic distortion.",
        evidence_refs=["honeycombing"],
        confidence=0.98,
        step_number=4,
    ))
    return board


@pytest.fixture
def clinical_attribution(clinical_deadlock_board: InMemoryBlackboard) -> AttributionResult:
    """Runs deadlock attribution on the clinical deadlock board."""
    engine = DeadlockAttributionEngine()
    return engine.analyze(clinical_deadlock_board, window=2)


@pytest.fixture
def clinical_sandbox(clinical_deadlock_board: InMemoryBlackboard, clinical_attribution: AttributionResult) -> SandboxSession:
    """Spawns an isolated sandbox session for the clinical deadlock."""
    mgr = SandboxManager()
    return mgr.create_session_from_attribution(clinical_deadlock_board, clinical_attribution)


def test_simulator_stops_on_instant_consensus(clinical_sandbox: SandboxSession, clinical_attribution: AttributionResult):
    """Iteration 1 score >= 0.85 -> CONSENSUS_FOUND."""
    simulator = CounterfactualSimulator(target_score=0.85, max_iterations=6)
    broker = MockLLMBroker(default_response={
        "tag": PXPTag.RATIFY.value,
        "prediction": "Idiopathic Pulmonary Fibrosis (IPF)",
        "explanation": "Synthesized consensus: HRCT subpleural honeycombing and normal BNP confirm IPF because fibrosis causes the crackles.",
        "confidence": 0.90,
        "evidence_refs": ["honeycombing", "normal_bnp", "crackles"],
    })

    result = simulator.run_adaptive_search(
        sandbox_session=clinical_sandbox,
        attribution=clinical_attribution,
        broker=broker,
    )

    assert result.status == SimulationStatus.CONSENSUS_FOUND
    assert result.total_iterations == 1
    assert result.best_score >= 0.85
    assert result.winning_candidate is not None
    assert result.winning_candidate.score >= 0.85
    assert result.winning_candidate.score_delta == 0.0
    assert len(result.score_trajectory) == 1
    assert result.plateau_proven is False
    assert "consensus" in result.reason.lower()


def test_simulator_improves_iteratively(clinical_sandbox: SandboxSession, clinical_attribution: AttributionResult, monkeypatch):
    """Climbing score 0.40 -> 0.65 -> 0.88 -> CONSENSUS_FOUND."""
    simulator = CounterfactualSimulator(target_score=0.85, max_iterations=6)

    score_sequence = [0.40, 0.65, 0.88]
    call_idx = 0

    def mock_score(revised, peer, ground_truth=None):
        nonlocal call_idx
        s = score_sequence[call_idx]
        call_idx += 1
        return s

    monkeypatch.setattr(simulator, "compute_agreement_score", mock_score)

    result = simulator.run_adaptive_search(
        sandbox_session=clinical_sandbox,
        attribution=clinical_attribution,
    )

    assert result.status == SimulationStatus.CONSENSUS_FOUND
    assert result.total_iterations == 3
    assert result.score_trajectory == [0.40, 0.65, 0.88]
    assert result.best_score == 0.88
    assert result.winning_candidate is not None
    assert result.winning_candidate.iteration == 3
    assert result.winning_candidate.score == 0.88
    assert result.winning_candidate.score_delta == pytest.approx(0.23, rel=1e-3)
    assert result.plateau_proven is False


def test_simulator_detects_plateau_and_stops_early(clinical_sandbox: SandboxSession, clinical_attribution: AttributionResult, monkeypatch):
    """Stagnant scores 0.30 -> 0.32 -> 0.31 -> stops at iteration 3 with PLATEAU_DETECTED."""
    simulator = CounterfactualSimulator(target_score=0.85, patience=2, epsilon=0.05, max_iterations=6)

    stagnant_scores = [0.30, 0.32, 0.31, 0.31, 0.31, 0.31]
    call_idx = 0

    def mock_score(revised, peer, ground_truth=None):
        nonlocal call_idx
        s = stagnant_scores[call_idx]
        call_idx += 1
        return s

    monkeypatch.setattr(simulator, "compute_agreement_score", mock_score)

    result = simulator.run_adaptive_search(
        sandbox_session=clinical_sandbox,
        attribution=clinical_attribution,
    )

    assert result.status == SimulationStatus.PLATEAU_DETECTED
    assert result.total_iterations == 3
    assert result.plateau_proven is True
    assert result.score_trajectory == [0.30, 0.32, 0.31]
    assert result.best_score == 0.32
    assert result.winning_candidate is not None
    assert result.winning_candidate.iteration == 2
    assert result.winning_candidate.score == 0.32
    assert "plateau" in result.reason.lower()


def test_simulator_respects_max_cap(clinical_sandbox: SandboxSession, clinical_attribution: AttributionResult, monkeypatch):
    """Score increases by tiny amounts < epsilon each round, hits max_iterations 6 -> BUDGET_EXHAUSTED."""
    # Set patience=10 so search doesn't terminate early on plateau
    simulator = CounterfactualSimulator(target_score=0.85, patience=10, epsilon=0.05, max_iterations=6)

    tiny_increase_scores = [0.40, 0.41, 0.42, 0.43, 0.44, 0.45]
    call_idx = 0

    def mock_score(revised, peer, ground_truth=None):
        nonlocal call_idx
        s = tiny_increase_scores[call_idx]
        call_idx += 1
        return s

    monkeypatch.setattr(simulator, "compute_agreement_score", mock_score)

    result = simulator.run_adaptive_search(
        sandbox_session=clinical_sandbox,
        attribution=clinical_attribution,
    )

    assert result.status == SimulationStatus.BUDGET_EXHAUSTED
    assert result.total_iterations == 6
    assert result.plateau_proven is False
    assert result.score_trajectory == [0.40, 0.41, 0.42, 0.43, 0.44, 0.45]
    assert result.best_score == 0.45
    assert result.winning_candidate is not None
    assert result.winning_candidate.iteration == 6
    assert result.winning_candidate.score == 0.45
    assert "exhausted" in result.reason.lower()


def test_simulation_preserves_sandbox_isolation(clinical_deadlock_board: InMemoryBlackboard, clinical_attribution: AttributionResult):
    """Live board entries remain unaffected while sandbox contains simulated entries."""
    mgr = SandboxManager()
    sandbox = mgr.create_session_from_attribution(clinical_deadlock_board, clinical_attribution)

    live_entries_before = clinical_deadlock_board.get_entries("main")
    count_before = len(live_entries_before)
    entry_ids_before = {e.entry_id for e in live_entries_before}

    simulator = CounterfactualSimulator(target_score=0.85, max_iterations=3)
    result = simulator.run_adaptive_search(sandbox, clinical_attribution)

    # 1. Live board remains pristine
    live_entries_after = clinical_deadlock_board.get_entries("main")
    assert len(live_entries_after) == count_before
    assert {e.entry_id for e in live_entries_after} == entry_ids_before
    assert all(e.branch_id == "main" for e in live_entries_after)
    assert not any(e.metadata.get("is_simulated") for e in live_entries_after)

    # 2. Sandbox contains simulated entries
    sim_entries = sandbox.get_simulated_entries()
    assert len(sim_entries) == result.total_iterations * 2
    assert all(e.branch_id == sandbox.sandbox_branch_id for e in sim_entries)
    assert all(e.metadata.get("is_simulated") is True for e in sim_entries)

    # 3. Telemetry graph exports simulated nodes and edges safely
    graph = sandbox.export_graph_for_telemetry()
    assert graph["sandbox_branch_id"] == sandbox.sandbox_branch_id
    sim_nodes = [n for n in graph["nodes"] if n["is_simulated"]]
    assert len(sim_nodes) == len(sim_entries)


def test_scoring_function_mechanics():
    """Validates tag weights, evidence overlap, overconfidence penalty, and bounds."""
    simulator = CounterfactualSimulator()

    base_revised = BlackboardEntry(
        entry_id="e_rev",
        branch_id="sandbox_test",
        agent_id="agent_cardio",
        tag=PXPTag.REVISE,
        prediction="Synthesized Hypothesis",
        explanation="Detailed clinical synthesis because evidence suggests unified pathology.",
        evidence_refs=["ev_a", "ev_b"],
        confidence=0.85,
    )

    # 1. Tag weight validation: RATIFY > REVISE > REFUTE > REJECT
    peer_ratify = BlackboardEntry(
        entry_id="p1", branch_id="sandbox_test", agent_id="agent_pulmo",
        tag=PXPTag.RATIFY, prediction="Synthesized Hypothesis",
        explanation="Detailed agreement because evidence indicates consistent pathology.",
        evidence_refs=["ev_a", "ev_b"], confidence=0.85,
    )
    peer_revise = BlackboardEntry(
        entry_id="p2", branch_id="sandbox_test", agent_id="agent_pulmo",
        tag=PXPTag.REVISE, prediction="Synthesized Hypothesis",
        explanation="Detailed partial modification because evidence indicates consistent pathology.",
        evidence_refs=["ev_a", "ev_b"], confidence=0.85,
    )
    peer_refute = BlackboardEntry(
        entry_id="p3", branch_id="sandbox_test", agent_id="agent_pulmo",
        tag=PXPTag.REFUTE, prediction="Conflicting Diagnosis",
        explanation="Detailed refutation because evidence indicates consistent pathology.",
        evidence_refs=["ev_a", "ev_b"], confidence=0.85,
    )
    peer_reject = BlackboardEntry(
        entry_id="p4", branch_id="sandbox_test", agent_id="agent_pulmo",
        tag=PXPTag.REJECT, prediction="Total Disagreement",
        explanation="Complete rejection because evidence indicates consistent pathology.",
        evidence_refs=["ev_a", "ev_b"], confidence=0.85,
    )

    s_ratify = simulator.compute_agreement_score(base_revised, peer_ratify)
    s_revise = simulator.compute_agreement_score(base_revised, peer_revise)
    s_refute = simulator.compute_agreement_score(base_revised, peer_refute)
    s_reject = simulator.compute_agreement_score(base_revised, peer_reject)

    assert s_ratify > s_revise > s_refute > s_reject

    # 2. Evidence overlap validation
    peer_identical_ev = BlackboardEntry(
        entry_id="p_ev1", branch_id="sandbox_test", agent_id="agent_pulmo",
        tag=PXPTag.RATIFY, prediction="Synthesized Hypothesis",
        explanation="Thorough confirmation because shared data is solid.",
        evidence_refs=["ev_a", "ev_b"], confidence=0.85,
    )
    peer_partial_ev = BlackboardEntry(
        entry_id="p_ev2", branch_id="sandbox_test", agent_id="agent_pulmo",
        tag=PXPTag.RATIFY, prediction="Synthesized Hypothesis",
        explanation="Thorough confirmation because shared data is solid.",
        evidence_refs=["ev_b", "ev_c"], confidence=0.85,
    )
    peer_disjoint_ev = BlackboardEntry(
        entry_id="p_ev3", branch_id="sandbox_test", agent_id="agent_pulmo",
        tag=PXPTag.RATIFY, prediction="Synthesized Hypothesis",
        explanation="Thorough confirmation because shared data is solid.",
        evidence_refs=["ev_z"], confidence=0.85,
    )
    peer_empty_ev = BlackboardEntry(
        entry_id="p_ev4", branch_id="sandbox_test", agent_id="agent_pulmo",
        tag=PXPTag.RATIFY, prediction="Synthesized Hypothesis",
        explanation="Thorough confirmation because shared data is solid.",
        evidence_refs=[], confidence=0.85,
    )
    rev_empty_ev = BlackboardEntry(
        entry_id="r_empty", branch_id="sandbox_test", agent_id="agent_cardio",
        tag=PXPTag.REVISE, prediction="Synthesized Hypothesis",
        explanation="Thorough confirmation because shared data is solid.",
        evidence_refs=[], confidence=0.85,
    )

    s_identical = simulator.compute_agreement_score(base_revised, peer_identical_ev)
    s_partial = simulator.compute_agreement_score(base_revised, peer_partial_ev)
    s_disjoint = simulator.compute_agreement_score(base_revised, peer_disjoint_ev)
    s_both_empty = simulator.compute_agreement_score(rev_empty_ev, peer_empty_ev)

    assert s_identical > s_partial > s_disjoint
    assert 0.0 <= s_both_empty <= 1.0

    # 3. Overconfidence penalty
    peer_overconfident_no_ev = BlackboardEntry(
        entry_id="p_oc", branch_id="sandbox_test", agent_id="agent_pulmo",
        tag=PXPTag.REFUTE, prediction="Opposing Position",
        explanation="Dogmatic refutation without referencing common facts.",
        evidence_refs=["unshared_1"], confidence=0.98,
    )
    peer_calibrated_no_ev = BlackboardEntry(
        entry_id="p_cal", branch_id="sandbox_test", agent_id="agent_pulmo",
        tag=PXPTag.REFUTE, prediction="Opposing Position",
        explanation="Dogmatic refutation without referencing common facts.",
        evidence_refs=["unshared_1"], confidence=0.80,
    )

    s_oc = simulator.compute_agreement_score(base_revised, peer_overconfident_no_ev)
    s_cal = simulator.compute_agreement_score(base_revised, peer_calibrated_no_ev)
    assert s_oc < s_cal  # Penalized for >0.9 confidence with no mutual evidence

    # 4. Strict bounds [0.0, 1.0]
    assert 0.0 <= s_reject <= 1.0
    assert 0.0 <= s_ratify <= 1.0
    assert 0.0 <= s_oc <= 1.0


def test_generate_synthesis_prompt():
    """Verifies prompt incorporates clashing evidence, positions, and prior critique."""
    simulator = CounterfactualSimulator()
    attribution = AttributionResult(
        deadlock_detected=True,
        conflicting_agents=["agent_cardio", "agent_pulmo"],
        contested_predictions={
            "agent_cardio": "Congestive Heart Failure (CHF)",
            "agent_pulmo": "Idiopathic Pulmonary Fibrosis (IPF)",
        },
        clashing_evidence=["crackles", "honeycombing", "normal_bnp"],
        reason="Circular dispute between cardio and pulmo over volume overload vs honeycombing.",
    )

    prompt_with_critique = simulator.generate_synthesis_prompt(
        attribution=attribution,
        target_agent_id="agent_cardio",
        prior_critique="Explain why BNP is normal if volume overload is present.",
    )

    assert "agent_cardio" in prompt_with_critique
    assert "Congestive Heart Failure (CHF)" in prompt_with_critique
    assert "Idiopathic Pulmonary Fibrosis (IPF)" in prompt_with_critique
    assert "crackles" in prompt_with_critique
    assert "honeycombing" in prompt_with_critique
    assert "normal_bnp" in prompt_with_critique
    assert "Explain why BNP is normal if volume overload is present." in prompt_with_critique
    assert "[PRIOR PEER CRITIQUE]" in prompt_with_critique

    prompt_no_critique = simulator.generate_synthesis_prompt(
        attribution=attribution,
        target_agent_id="agent_cardio",
        prior_critique=None,
    )
    assert "agent_cardio" in prompt_no_critique
    assert "[PRIOR PEER CRITIQUE]" not in prompt_no_critique


def test_compute_agreement_score_with_ground_truth_bonus():
    """Validates that predictions/explanations aligning with ground truth receive the +0.10 quality bonus."""
    simulator = CounterfactualSimulator()

    base_rev = BlackboardEntry(
        entry_id="r1",
        branch_id="sandbox_test",
        agent_id="agent_cardio",
        tag=PXPTag.REVISE,
        prediction="Idiopathic Pulmonary Fibrosis (IPF)",
        explanation="Clinical reasoning confirms IPF based on evidence.",
        evidence_refs=["honeycombing"],
        confidence=0.85,
    )
    peer_reaction = BlackboardEntry(
        entry_id="p1",
        branch_id="sandbox_test",
        agent_id="agent_pulmo",
        tag=PXPTag.RATIFY,
        prediction="Idiopathic Pulmonary Fibrosis (IPF)",
        explanation="Complete agreement confirms IPF diagnosis.",
        evidence_refs=["honeycombing"],
        confidence=0.85,
    )

    score_with_gt = simulator.compute_agreement_score(base_rev, peer_reaction, ground_truth="Idiopathic Pulmonary Fibrosis")
    score_without_gt = simulator.compute_agreement_score(base_rev, peer_reaction, ground_truth=None)

    assert score_with_gt > score_without_gt
    assert pytest.approx(score_with_gt - score_without_gt, abs=0.03) == 0.025  # 0.10 * w_expl (0.25)


def test_simulator_handles_malformed_broker_payload(clinical_sandbox: SandboxSession, clinical_attribution: AttributionResult):
    """Verifies that simulator does not crash on empty strings, None confidence, or missing keys from broker."""
    simulator = CounterfactualSimulator(target_score=0.85, max_iterations=2)
    # Broker returning empty/malformed responses
    malformed_broker = MockLLMBroker(default_response={
        "prediction": "   ",  # empty whitespace string
        "explanation": "",     # empty string
        "confidence": None,    # None confidence
        "evidence_refs": None, # None evidence
        "metadata": None,      # None metadata
    })

    result = simulator.run_adaptive_search(
        sandbox_session=clinical_sandbox,
        attribution=clinical_attribution,
        broker=malformed_broker,
    )

    # Should run without crashing
    assert result.total_iterations >= 1
    # Verify simulated entries in sandbox were sanitized with fallback non-empty strings
    sim_entries = clinical_sandbox.get_simulated_entries()
    assert len(sim_entries) > 0
    for e in sim_entries:
        assert len(e.prediction.strip()) > 0
        assert len(e.explanation.strip()) > 0
        assert 0.0 <= e.confidence <= 1.0
        assert isinstance(e.evidence_refs, list)
        assert isinstance(e.metadata, dict)


def test_simulator_opposing_agent_resolution_avoids_collision(clinical_sandbox: SandboxSession, clinical_attribution: AttributionResult):
    """Verifies that explicitly specifying target_agent_id as the 2nd agent selects the 1st as opponent, avoiding self-debate."""
    simulator = CounterfactualSimulator(target_score=0.85, max_iterations=1)

    # conflicting_agents = ["agent_cardio", "agent_pulmo"]
    # If target_agent_id is explicitly passed as "agent_pulmo"
    result = simulator.run_adaptive_search(
        sandbox_session=clinical_sandbox,
        attribution=clinical_attribution,
        target_agent_id="agent_pulmo",
        opposing_agent_id=None,
    )

    sim_entries = clinical_sandbox.get_simulated_entries()
    assert len(sim_entries) == 2
    revised_entry = sim_entries[0]
    reaction_entry = sim_entries[1]

    # Target is pulmo, opponent MUST NOT be pulmo (must be cardio)
    assert revised_entry.agent_id == "agent_pulmo"
    assert reaction_entry.agent_id == "agent_cardio"
    assert revised_entry.agent_id != reaction_entry.agent_id


def test_simulator_handles_single_agent_conflict():
    """Verifies simulator handles attribution with only 1 conflicting agent gracefully."""
    board = InMemoryBlackboard(session_id="single_agent_sess", task_id="task_1", problem_statement="Problem")
    board.add_entry(BlackboardEntry(
        entry_id="e1", branch_id="main", agent_id="agent_solo",
        tag=PXPTag.PROPOSE, prediction="Hypothesis", explanation="Sole agent thinking.",
        step_number=1,
    ))
    attribution = AttributionResult(
        deadlock_detected=True,
        conflicting_agents=["agent_solo"],
        divergence_entry_id="e1",
        rollback_step=1,
    )
    mgr = SandboxManager()
    sandbox = mgr.create_session_from_attribution(board, attribution)

    simulator = CounterfactualSimulator(target_score=0.85, max_iterations=1)
    result = simulator.run_adaptive_search(sandbox, attribution)

    assert result.total_iterations == 1
    sim_entries = sandbox.get_simulated_entries()
    assert len(sim_entries) == 2
    assert sim_entries[0].agent_id == "agent_solo"
    assert sim_entries[1].agent_id == "agent_opposing"


def test_simulator_supports_llm_response_and_agent_contribution(clinical_sandbox: SandboxSession, clinical_attribution: AttributionResult):
    """Verifies that simulator seamlessly unpacks broker responses containing .text attributes (Student 2 LLMResponse format)."""
    class FakeLLMResponse:
        def __init__(self, text: str):
            self.text = text

    class FakeModelBroker:
        def generate(self, req):
            payload = json.dumps({
                "tag": "RATIFY",
                "prediction": "Reconciled clinical diagnosis",
                "explanation": "Harmonized Honeycombing with elevated BNP findings because clinical evidence confirms consistent findings across both pulmonary and cardiac panels.",
                "evidence_refs": ["HRCT-01", "BNP-900"],
                "confidence": 0.88,
            })
            return FakeLLMResponse(text=payload)

    broker = FakeModelBroker()
    simulator = CounterfactualSimulator(target_score=0.85, max_iterations=1)
    result = simulator.run_adaptive_search(
        sandbox_session=clinical_sandbox,
        attribution=clinical_attribution,
        broker=broker,
    )
    assert result.status == SimulationStatus.CONSENSUS_FOUND
    assert result.best_score >= 0.85
    sim_entries = clinical_sandbox.get_simulated_entries()
    assert len(sim_entries) == 2
    assert "Reconciled clinical diagnosis" in sim_entries[0].prediction


