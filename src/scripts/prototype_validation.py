"""Exact declared-calculation checks and current-proposal votes for the prototype.

No task ids, question-specific solvers, references, eval, or executable model code.
Arithmetic consistency is weaker than task correctness: the model still chooses
the expression, and may choose the wrong one for the question.
"""
from __future__ import annotations

import ast
from fractions import Fraction
import re

from agents.base import PEXAgent
from contracts.schemas import PXPTag
from prompts.validation import DecisionSemanticError, normalize_claim

VALIDATION_POLICY = "exact-arithmetic-and-current-proposal-v3"
CALC_INSTRUCTIONS = (
    "For every purely numerical prediction.claim, include at least one evidence string "
    "formatted CALC: <arithmetic expression> = <result>. Use only numbers, parentheses, "
    "+, -, *, /; parenthesize numerator and denominator fractions before dividing them. "
    "A chain such as CALC: expression = intermediate expression = result is allowed, "
    "but every equality must be true. A bare CALC: expression is also accepted; the calculator "
    "computes its value. Square grouping brackets and the symbols ×, ÷, − are supported. "
    "Keep symbolic labels, variables and prose outside CALC entries: substitute the numbers, "
    "e.g. write CALC: (12 - 4) / 2 = 4 rather than CALC: 2x = 8 -> x = 4. "
    "Numeric equality steps elsewhere in evidence will also be checked, including "
    "steps following a symbolic label. Show normalization sums before substituting "
    "their values into the final division. Do not present false equalities as evidence; "
    "describe rejected alternatives in prose instead. "
    "The expression must derive the answer from the task's quantities, not merely repeat "
    "the answer. The last CALC result must equal prediction.claim exactly. "
    "For nonnumeric answers, include CALC entries for any supporting arithmetic. "
    "A RATIFY restates a proposal whose CALC evidence was already checked, so it needs no new "
    "CALC entry; any arithmetic you do include is still checked. "
    "All CALC equalities will be checked with an exact rational calculator before posting. "
    "False calculations must be corrected, not copied. This checks arithmetic, not assumptions. "
    "Keep the rationale concise so there is room for the CALC evidence in the JSON. "
)
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*/\s*[+-]?\d+)?\Z")


def exact_value(expression: str) -> Fraction:
    """Evaluate a bounded arithmetic AST; never execute Python from the model."""
    expression = expression.strip().translate(str.maketrans({"[": "(", "]": ")", "×": "*", "÷": "/", "−": "-"}))
    if not expression or len(expression) > 512:
        raise ValueError("expression must contain 1–512 characters")
    if not re.fullmatch(r"[0-9.\s+*/()\-]+", expression):
        raise ValueError("use only numbers, parentheses, +, -, *, /")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, RecursionError) as exc:
        raise ValueError("invalid arithmetic syntax") from exc
    if sum(1 for _ in ast.walk(tree)) > 128:
        raise ValueError("expression is too complex")

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            token = ast.get_source_segment(expression, node)
            if len(token) > 60:
                raise ValueError("numeric literal is too long")
            result = Fraction(token)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            result = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): result = left + right
            elif isinstance(node.op, ast.Sub): result = left - right
            elif isinstance(node.op, ast.Mult): result = left * right
            else: result = left / right
        else:
            raise ValueError("unsupported arithmetic operator")
        if max(result.numerator.bit_length(), result.denominator.bit_length()) > 4096:
            raise ValueError("arithmetic result is too large")
        return result

    try:
        return visit(tree)
    except (ZeroDivisionError, RecursionError) as exc:
        raise ValueError("division by zero or excessive nesting") from exc


def checked_calculations(decision):
    # A RATIFY must restate the current proposal's claim exactly, and that proposal's
    # CALC evidence was checked when it was posted. Re-proving it is not required, and
    # algebraic working (2x = 0.10) in a ratification is a label, not an error. Any
    # numeric equality it does state is still checked.
    endorsement = decision.tag == PXPTag.RATIFY
    checks, supporting_checks = [], []
    for entry in decision.explanation.evidence:
        prefixed = entry.strip().upper().startswith("CALC:")
        explicit = prefixed and not endorsement
        if not explicit and "=" not in entry:
            continue
        equation = entry.strip()[5:].strip() if prefixed else entry.strip()
        try:
            parts = [part.strip() for part in equation.split("=")]
            if len(parts) > 16 or len(equation) > 4096:
                if not explicit:
                    continue
                raise ValueError("a CALC entry may contain at most 16 expressions and 4096 characters")
            values = []
            for part in parts:
                try:
                    values.append(exact_value(part))
                except ValueError:
                    if explicit:
                        raise
                    # Labels such as P(D2|H3) are not numeric expressions. Audit
                    # the adjacent numeric steps without interpreting the label.
                    values.append(None)
        except ValueError as exc:
            raise DecisionSemanticError("invalid_calculation", f"Invalid CALC entry: {exc}. Use numeric expressions only, e.g. CALC: (numeric expression) = result. Omit symbolic labels and prose.") from exc
        if explicit and len(parts) == 1:
            checks.append({"expression": parts[0], "value": str(values[0])})
        for index in range(len(parts) - 1):
            expression, stated = parts[index:index + 2]
            actual, expected = values[index:index + 2]
            if actual is None or expected is None:
                continue
            if actual != expected:
                raise DecisionSemanticError("arithmetic_mismatch",
                    f"Exact calculator: {expression} = {actual}, not {stated}. "
                    "Recheck your calculation, rationale and prediction.claim, then choose the appropriate tag and current target. "
                    "The calculator verifies the expression you supplied; it does not choose the expression for you.")
            (checks if explicit else supporting_checks).append({"expression": expression, "value": str(actual)})
    claim = normalize_claim(decision.prediction.claim)
    if _NUMBER.fullmatch(claim) and not endorsement:
        if not checks:
            raise DecisionSemanticError("missing_calculation",
                "A numerical prediction needs evidence containing CALC: <expression using the task quantities> = <result>. "
                "Include the final computation, with fractions parenthesized; the last result must equal prediction.claim. "
                "Write numbers, not variables: e.g. CALC: (12 - 4) / 2 = 4, not CALC: 2x = 8 -> x = 4.")
        try:
            claim_value = exact_value(claim)
        except ValueError as exc:
            raise DecisionSemanticError("invalid_numeric_claim", str(exc)) from exc
        if claim_value != Fraction(checks[-1]["value"]):
            raise DecisionSemanticError("calculation_claim_mismatch",
                f"Your last verified CALC result is {checks[-1]['value']} but prediction.claim is {claim}. "
                "Resolve this contradiction in the calculation, rationale and claim; do not change a correct expression merely to match an old claim.")
    return supporting_checks + checks


class PrototypeAgent(PEXAgent):
    """Reprompt on invalid arithmetic/votes; never silently replace model answers."""
    def spec(self):
        return {**super().spec(), "validation_policy": VALIDATION_POLICY}

    def validate_output(self, decision, snapshot):
        decision, normalizations = super().validate_output(decision, snapshot)
        if decision.tag == PXPTag.RATIFY:
            active = next((c for c in reversed(snapshot.contributions) if c.tag == PXPTag.REVISE), None)
            by_id = {c.contribution_id: c for c in snapshot.contributions}
            target = by_id.get(decision.target_contribution_id)
            seen = set()
            while target is not None and target.tag == PXPTag.RATIFY and target.contribution_id not in seen:
                seen.add(target.contribution_id)
                target = by_id.get(target.target_contribution_id)
            if active is None or target is None or target.contribution_id != active.contribution_id or normalize_claim(decision.prediction.claim) != normalize_claim(active.prediction):
                raise DecisionSemanticError("stale_ratification",
                    f"RATIFY must endorse the current proposal id={active.contribution_id if active else 'none'}, "
                    f"claim={active.prediction if active else 'none'}, not a superseded proposal or criticism. "
                    "Check its arithmetic and rationale; RATIFY that id only if you agree, otherwise REVISE or criticize it.")
        checked_calculations(decision)
        return decision, normalizations

    def to_contribution(self, snapshot, decision, *args, **kwargs):
        kwargs["validation_policy"] = VALIDATION_POLICY
        kwargs["arithmetic_checks"] = checked_calculations(decision)
        return super().to_contribution(snapshot, decision, *args, **kwargs)
