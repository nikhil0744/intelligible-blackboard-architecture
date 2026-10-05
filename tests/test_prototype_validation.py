"""False arithmetic and obsolete votes must be repaired before commitment."""
from fractions import Fraction
import json

import pytest

from agents.base import AgentTurnError, PEXAgent
from agents.board_client import InMemoryBoard
from llm_broker import ModelBroker
from llm_broker.backends.mock import MockBackend
from prompts.schema import AgentDecision
from prompts.validation import DecisionSemanticError
from scripts.prototype_validation import PrototypeAgent, checked_calculations, exact_value


def output(claim="10/19", evidence=None, tag="REVISE", target=None):
    return {"tag": tag, "target_contribution_id": target,
        "prediction": {"claim": claim, "confidence": .8},
        "explanation": {"rationale": "Calculate the observation weights and normalize.",
            "evidence": evidence if evidence is not None else ["CALC: (1/3) / ((9/10)*(1/3) + (1/3)) = 10/19"]}}


@pytest.mark.parametrize("expression,expected", [
    ("(1/3) / ((9/10)*(1/3) + (1/3))", Fraction(10, 19)),
    ("(1/3) / (19/30)", Fraction(10, 19)),
    ("1 - (9/10 * 1/3) / (19/30)", Fraction(10, 19)),
    ("(1.10 - 1.00) / 2", Fraction(1, 20)),
    ("0.1 + 0.2", Fraction(3, 10)),
    ("-(-.5) + (+2)", Fraction(5, 2)),
    ("(1/3) ÷ [(9/10) × (1/3) + (1/3)]", Fraction(10, 19)),
    ("1 − .5", Fraction(1, 2)),
])
def test_exact_arithmetic(expression, expected):
    assert exact_value(expression) == expected


@pytest.mark.parametrize("expression", [
    "__import__('os').system('echo unsafe')", "1/0", "2**100000", "1//2",
    "1e100", "[1][0]", "(1).__class__", "x + 1", "1 % 2", "1 << 2",
    "9" * 61, "1+" * 100 + "1", "(" * 300 + "1" + ")" * 300,
])
def test_unsupported_and_unbounded_expressions_are_rejected(expression):
    with pytest.raises(ValueError):
        exact_value(expression)


@pytest.mark.parametrize("claim,evidence,code", [
    ("1/19", ["CALC: (1/3) / (19/30) = 1/19"], "arithmetic_mismatch"),
    ("16/19", ["CALC: (9/10 * 1/3) / (19/30) = 3/19", "CALC: 1 - 3/19 = 16/19"], "arithmetic_mismatch"),
    ("1/19", ["CALC: (1/3) / (19/30) = 10/19"], "calculation_claim_mismatch"),
    ("1/19.", ["CALC: (1/3) / (19/30) = 10/19"], "calculation_claim_mismatch"),
    ("10/19", [], "missing_calculation"),
    ("10/19", ["CALC: f(3) = 10/19"], "invalid_calculation"),
    ("10/19", ["CALC: 1/3 = 19/30 = 10/19"], "arithmetic_mismatch"),
    ("10/19", ["CALC: (1/3)/(19/30) = 1/19 = 10/19"], "arithmetic_mismatch"),
    ("10/19", ["CALC: (1/3)/(19/30) = 10/19 = 1/19"], "arithmetic_mismatch"),
    ("1/19", ["CALC: (1/3)/(19/30)"], "calculation_claim_mismatch"),
    ("10/19", ["CALC: (1/3)/(19/30) ="], "invalid_calculation"),
    ("10/19", ["CALC: 10/19 == 10/19"], "invalid_calculation"),
    ("10/19", ["CALC: " + " = ".join(["10/19"] * 17)], "invalid_calculation"),
])
def test_false_or_missing_evidence_is_not_accepted(claim, evidence, code):
    with pytest.raises(DecisionSemanticError) as exc:
        checked_calculations(AgentDecision.model_validate(output(claim, evidence)))
    assert exc.value.code == code


def test_calculator_checks_consistency_not_the_question_answer():
    # A numerically valid but inappropriate expression can still be task-wrong.
    decision = AgentDecision.model_validate(output("2/3", ["CALC: 2 / 3 = 2/3"]))
    assert checked_calculations(decision)[0]["value"] == "2/3"
    assert checked_calculations(AgentDecision.model_validate(output("A", []))) == []


@pytest.mark.parametrize("evidence", [
    "CALC: (1/3) / ((9/10)*(1/3) + (1/3)) = (1/3) / (19/30) = 10/19",
    "CALC: (1/3) ÷ [(9/10) × (1/3) + (1/3)] = (1/3) ÷ (19/30) = 10/19",
    "CALC: (1/3) / (19/30)",
])
def test_valid_chains_and_bare_calculations_are_accepted_without_repairs(evidence):
    backend = MockBackend(responses=[json.dumps(output(evidence=[evidence]))])
    agent = PrototypeAgent("a", "general_reasoner", ModelBroker(backend, "test"))
    result = agent.act(InMemoryBoard().create_session("t", "Q"))
    assert result.prediction == "10/19" and result.metadata["parse_attempts"] == 1
    assert result.metadata["arithmetic_checks"][-1]["value"] == "10/19"
    assert result.metadata["repair_reasons"] == []
    assert len(result.metadata["arithmetic_checks"]) == (2 if "=" in evidence else 1)


@pytest.mark.parametrize("claim,evidence", [
    ("1/19", [
        "P(D2|H3) = (1/3 * 1) / [(1/3 * 9/10) + (1/3 * 1) + (1/3 * 0)] = 1/19",
        "CALC: 1/19 = 1/19",
    ]),
    ("5/6", [
        "P(D2|H3) = (1/3 * 1) / (9/30 + 1/3) = (1/3) / (12/30) = 10/12 = 5/6",
        "CALC: (1/3) / (12/30) = 10/12 = 5/6",
    ]),
    ("5/6", ["P(H3) = 9/30 + 1/3 = 12/30", "CALC: (1/3) / (12/30) = 5/6"]),
])
def test_actual_trace_errors_cannot_hide_behind_a_valid_final_calc(claim, evidence):
    with pytest.raises(DecisionSemanticError) as exc:
        checked_calculations(AgentDecision.model_validate(output(claim, evidence)))
    assert exc.value.code == "arithmetic_mismatch"


def test_correct_labelled_evidence_is_checked_without_binding_claim_to_supporting_sum():
    data = output(evidence=[
        "P(D1) = P(D2) = P(D3) = 1/3",
        "CALC: (1/3) / (19/30) = 10/19",
        "P(H3) = 9/30 + 1/3 = 19/30",
        "P(D2|H3) = (1/3) / (19/30) = 10/19",
    ])
    checks = checked_calculations(AgentDecision.model_validate(data))
    assert [check["value"] for check in checks] == ["19/30", "10/19", "10/19"]


@pytest.mark.parametrize("expression,claim", [
    ("1/3 / (9/30 + 10/30) = 1/3 / 19/30 = 10/19", "10/19"),
    ("1/3 / (9/30 + 10/30) = 1/3 /19/30 = 10/19", "10/19"),
    ("(1/3) / 19/30 = (1/3)/(19/30) = 10/19", "10/19"),
    ("1/3 ÷ 19/30 = (1/3) ÷ (19/30) = 10/19", "10/19"),
    ("2/5 / 3/7 = (2/5) / (3/7) = 14/15", "14/15"),
    ("6/2 / 3/4 = 4", "4"),
])
def test_fraction_operands_are_disambiguated_from_the_whole_chain(expression, claim):
    data = output(claim, ["CALC: " + expression])
    backend = MockBackend(responses=[json.dumps(data)])
    agent = PrototypeAgent("a", "general_reasoner", ModelBroker(backend, "test"))
    result = agent.act(InMemoryBoard().create_session("t", "Q"))
    assert result.prediction == claim and result.metadata["parse_attempts"] == 1
    assert result.metadata["raw_responses"] == [json.dumps(data)]
    assert result.metadata["normalizations"] == ["fraction_operands_disambiguated_from_chain"]
    assert any("normalized_expression" in c or "normalized_next_expression" in c
               for c in result.metadata["arithmetic_checks"])


def test_disambiguation_also_checks_labelled_evidence():
    data = output(evidence=[
        "P(D2|H3) = 1/3 / (9/30 + 10/30) = 1/3 / 19/30 = 10/19",
        "CALC: (1/3)/(19/30) = 10/19",
    ])
    assert all(c["value"] == "10/19" for c in checked_calculations(AgentDecision.model_validate(data)))


@pytest.mark.parametrize("claim,evidence,code", [
    ("1/19", "1/3 / 19/30 = 1/19", "arithmetic_mismatch"),
    ("1/19", "1/3 / (9/30 + 10/30) = 1/3 / 19/30 = 10/19", "calculation_claim_mismatch"),
    ("10/19", "1/3 / 19/30 = 10/19 = 1/1710", "arithmetic_mismatch"),
    ("10/19", "1/3/19/30 = 10/19", "arithmetic_mismatch"),
    ("10/19", "1/3 / 19/30", "ambiguous_fraction_notation"),
])
def test_disambiguation_does_not_fit_false_equalities_or_prediction(claim, evidence, code):
    with pytest.raises(DecisionSemanticError) as exc:
        checked_calculations(AgentDecision.model_validate(output(claim, ["CALC: " + evidence])))
    assert exc.value.code == code


def test_explicit_and_standard_division_remain_unchanged():
    assert exact_value("1/3 / 19/30") == Fraction(1, 1710)
    data = output("1/1710", ["CALC: 1/3 / 19/30 = 1/1710"])
    checks = checked_calculations(AgentDecision.model_validate(data))
    assert checks == [{"expression": "1/3 / 19/30", "value": "1/1710"}]
    # Explicitly grouped input cannot be reinterpreted to fit a wrong result.
    with pytest.raises(DecisionSemanticError):
        checked_calculations(AgentDecision.model_validate(output("1/1710", ["CALC: (1/3) / (19/30) = 1/1710"])))


def test_false_equality_is_reprompted_and_raw_evidence_preserved():
    bad = output("1/19", ["CALC: (1/3) / (19/30) = 1/19"])
    backend = MockBackend(responses=[json.dumps(bad), json.dumps(output())])
    agent = PrototypeAgent("a", "general_reasoner", ModelBroker(backend, "test"))
    snapshot = InMemoryBoard().create_session("unseen-task", "Compute from the given observation.")
    result = agent.act(snapshot)
    assert result.prediction == "10/19" and not snapshot.contributions
    assert result.metadata["parse_attempts"] == 2
    assert result.metadata["repair_reasons"][0].startswith("arithmetic_mismatch")
    assert result.metadata["arithmetic_checks"][0]["value"] == "10/19"
    assert len(result.metadata["raw_responses"]) == 2
    assert "= 10/19, not 1/19" in backend.calls[1].messages[-1].content


def test_persistent_bad_arithmetic_fails_without_silent_answer_replacement():
    bad = output("1/19", ["CALC: (1/3) / (19/30) = 1/19"])
    backend = MockBackend(responses=[json.dumps(bad)])
    agent = PrototypeAgent("a", "general_reasoner", ModelBroker(backend, "test"))
    snapshot = InMemoryBoard().create_session("t", "Q")
    with pytest.raises(AgentTurnError) as exc:
        agent.act(snapshot)
    assert exc.value.attempts == 3 and not snapshot.contributions
    assert all(reason.startswith("arithmetic_mismatch") for reason in exc.value.repair_reasons)


def ratify_after_checked_proposal(evidence):
    board = InMemoryBoard()
    sid = board.create_session("t", "Q").session_id
    proposer = PrototypeAgent("p", "general_reasoner", ModelBroker(MockBackend(responses=[json.dumps(
        output("0.05", ["CALC: (1.10 - 1.00) / 2 = 0.05"]))]), "test"))
    board.submit(proposer.act(board.get_snapshot(sid)))
    snapshot = board.get_snapshot(sid)
    vote = output("0.05", evidence, tag="RATIFY", target=snapshot.contributions[0].contribution_id)
    return PrototypeAgent("v", "general_critic", ModelBroker(MockBackend(responses=[json.dumps(vote)]), "test")), snapshot


@pytest.mark.parametrize("evidence", [
    # Observed live Qwen ratifications that previously exhausted all attempts.
    ["The total cost is x + (x + 1.00) = 1.10.", "Dividing by 2, x = 0.05."],
    ["Simplifying, 2x + 1.00 = 1.10.", "CALC: 2x + 1.00 = 1.10 -> 2x = 0.10 -> x = 0.05"],
    ["CALC: (0.10 / 2) = 0.05"],
])
def test_ratification_of_a_checked_proposal_need_not_reprove_it(evidence):
    agent, snapshot = ratify_after_checked_proposal(evidence)
    result = agent.act(snapshot)
    assert result.tag.value == "RATIFY" and result.prediction == "0.05"
    assert result.metadata["repair_reasons"] == []


@pytest.mark.parametrize("evidence", [["CALC: (1.10 - 1.00) / 2 = 0.5"], ["0.10 / 2 = 0.5"]])
def test_false_arithmetic_in_a_ratification_is_still_rejected(evidence):
    agent, snapshot = ratify_after_checked_proposal(evidence)
    with pytest.raises(AgentTurnError) as exc:
        agent.act(snapshot)
    assert all(reason.startswith("arithmetic_mismatch") for reason in exc.value.repair_reasons)


def test_new_numerical_proposals_still_require_numeric_calc():
    algebra = output("0.05", ["CALC: 2x + 1.00 = 1.10 -> 2x = 0.10 -> x = 0.05"])
    with pytest.raises(DecisionSemanticError) as exc:
        checked_calculations(AgentDecision.model_validate(algebra))
    assert exc.value.code == "invalid_calculation"


def test_legacy_agents_keep_their_original_validation_contract():
    backend = MockBackend(responses=[json.dumps(output("5", []))])
    agent = PEXAgent("a", "general_reasoner", ModelBroker(backend, "test"))
    assert agent.act(InMemoryBoard().create_session("t", "Q")).prediction == "5"
