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
    "Keep symbolic labels and prose outside CALC entries. "
    "Compact fraction operands separated by a spaced division, such as 1/3 / 19/30, "
    "can be clarified from an unambiguous neighbouring equality. Explicit parentheses "
    "are preferred. Ambiguous notation alone is never resolved from prediction.claim. "
    "Numeric equality steps elsewhere in evidence will also be checked, including "
    "steps following a symbolic label. Show normalization sums before substituting "
    "their values into the final division. Do not present false equalities as evidence; "
    "describe rejected alternatives in prose instead. "
    "The expression must derive the answer from the task's quantities, not merely repeat "
    "the answer. The last CALC result must equal prediction.claim exactly. "
    "For nonnumeric answers, include CALC entries for any supporting arithmetic. "
    "All CALC equalities will be checked with an exact rational calculator before posting. "
    "False calculations must be corrected, not copied. This checks arithmetic, not assumptions. "
    "Keep the rationale concise so there is room for the CALC evidence in the JSON. "
)
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*/\s*[+-]?\d+)?\Z")
_COMPACT_FRACTION = re.compile(r"(?<![\w.])(?:\d+(?:\.\d*)?|\.\d+)/(?:\d+(?:\.\d*)?|\.\d+)(?![\w.])")


def fraction_operands(expression: str) -> str:
    """Parenthesize compact fractions separated by spaces/operators.

    Preserve fully compact slash sequences such as 1/3/19/30: there is no
    lexical evidence of which slashes separate fraction operands.
    """
    expression = expression.strip().translate(str.maketrans({"×": "*", "÷": "/", "−": "-"}))
    spans = []
    for match in _COMPACT_FRACTION.finditer(expression):
        start, end = match.span()
        if start and expression[start - 1] == "/" and (start < 2 or not expression[start - 2].isspace()):
            continue
        if end < len(expression) and expression[end] == "/" and (end + 1 == len(expression) or not expression[end + 1].isspace()):
            continue
        spans.append((start, end))
    for start, end in reversed(spans):
        expression = expression[:start] + "(" + expression[start:end] + ")" + expression[end:]
    return expression


def expression_options(expression: str):
    """Standard arithmetic plus a bounded compact-fraction interpretation."""
    primary = exact_value(expression)
    options = {primary: expression}
    grouped = fraction_operands(expression)
    if grouped != expression:
        try:
            alternative = exact_value(grouped)
        except ValueError:
            # Disambiguation cannot bypass the calculator's bounds/operators.
            pass
        else:
            if alternative != primary:
                options[alternative] = grouped
    return primary, options


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
    checks, supporting_checks = [], []
    for entry in decision.explanation.evidence:
        explicit = entry.strip().upper().startswith("CALC:")
        if not explicit and "=" not in entry:
            continue
        equation = entry.strip()[5:].strip() if explicit else entry.strip()
        try:
            parts = [part.strip() for part in equation.split("=")]
            if len(parts) > 16 or len(equation) > 4096:
                if not explicit:
                    continue
                raise ValueError("a CALC entry may contain at most 16 expressions and 4096 characters")
            values, options = [], []
            for part in parts:
                try:
                    primary, alternatives = expression_options(part)
                    values.append(primary)
                    options.append(alternatives)
                except ValueError:
                    if explicit:
                        raise
                    # Labels such as P(D2|H3) are not numeric expressions. Audit
                    # the adjacent numeric steps without interpreting the label.
                    values.append(None)
                    options.append(None)
        except ValueError as exc:
            raise DecisionSemanticError("invalid_calculation", f"Invalid CALC entry: {exc}. Use numeric expressions only, e.g. CALC: (numeric expression) = result. Omit symbolic labels and prose.") from exc
        # A single, unambiguous value must anchor each numeric equality run.
        # Resolve the whole run together, never fit each pair independently or
        # select a grouping from the final prediction/reference answer.
        resolved = list(parts)
        start = 0
        while start < len(parts):
            if options[start] is None:
                start += 1
                continue
            end = start + 1
            while end < len(parts) and options[end] is not None:
                end += 1
            candidates = set(options[start])
            for alternatives in options[start + 1:end]:
                candidates.intersection_update(alternatives)
            if len(candidates) == 1 and any(len(alternatives) == 1 for alternatives in options[start:end]):
                chosen = next(iter(candidates))
                for index in range(start, end):
                    values[index] = chosen
                    resolved[index] = options[index][chosen]
            start = end
        if explicit and len(parts) == 1:
            if len(options[0]) > 1:
                raise DecisionSemanticError("ambiguous_fraction_notation",
                    "A bare CALC has ambiguous fraction grouping. Parenthesize both fraction operands "
                    "or include an unambiguous neighbouring equality. Do not choose a grouping merely to match prediction.claim.")
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
            check = {"expression": expression, "value": str(actual)}
            if resolved[index] != expression:
                check["normalized_expression"] = resolved[index]
            if resolved[index + 1] != stated:
                check["normalized_next_expression"] = resolved[index + 1]
            (checks if explicit else supporting_checks).append(check)
    claim = normalize_claim(decision.prediction.claim)
    if _NUMBER.fullmatch(claim):
        if not checks:
            raise DecisionSemanticError("missing_calculation",
                "A numerical prediction needs evidence containing CALC: <expression using the task quantities> = <result>. "
                "Include the final computation, with fractions parenthesized; the last result must equal prediction.claim.")
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
        checks = checked_calculations(decision)
        if any("normalized_expression" in c or "normalized_next_expression" in c for c in checks):
            normalizations.append("fraction_operands_disambiguated_from_chain")
        return decision, normalizations

    def to_contribution(self, snapshot, decision, *args, **kwargs):
        kwargs["validation_policy"] = VALIDATION_POLICY
        kwargs["arithmetic_checks"] = checked_calculations(decision)
        return super().to_contribution(snapshot, decision, *args, **kwargs)
