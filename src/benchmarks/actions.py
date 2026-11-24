"""Benchmark actions extend PEXAgent's validated inference path, not a second agent loop."""
from __future__ import annotations

import json
import math
import re
from decimal import Decimal, InvalidOperation
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

from agents.base import PEXAgent
from contracts.schemas import PXPTag
from prompts import AgentDecision, DecisionParseError, DecisionSemanticError
from prompts.budget import build_prompt
from prompts.schema import DecisionExplanation, DecisionPrediction
from prompts.validation import normalize_claim, validate_decision

from .data_tools import TOOL_HELP


NUMERIC_ANSWER = re.compile(r"[+-]?(?:(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
TERMINAL_ANSWER = re.compile(rf"\b(?:is|equals)\s+({NUMERIC_ANSWER.pattern})[.!]?\s*$", re.IGNORECASE)
YEAR_QUALIFIER = re.compile(
    r"\b(?:between\s+[12][0-9]{3}\s+and\s+[12][0-9]{3}"
    r"|from\s+[12][0-9]{3}\s+(?:to|through)\s+[12][0-9]{3}"
    r"|(?:in|during|for)\s+[12][0-9]{3})\b",
    re.IGNORECASE,
)
NUMERIC_TYPES = {"numeric_exact", "numeric_approximate"}
ANSWER_DESCRIPTIONS = {
    "numeric_exact": "a JSON number without units or thousands separators",
    "numeric_approximate": "a JSON number rounded as requested, without units or thousands separators",
    "string_exact": "a JSON string containing only the requested text (such as a category, Yes, No, True, or False)",
    "list_exact": "a JSON array of the requested numbers or names, in the requested order",
}
Answer = int | float | str | list[int | float | str]


def numeric_answer(value) -> tuple[str, Decimal]:
    """Accept a single number, optionally with ordinary thousands separators."""
    text = str(value).strip()
    if not NUMERIC_ANSWER.fullmatch(text):
        raise ValueError("expected one number with no units or other text")
    text = text.replace(",", "")
    try:
        number = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError("invalid numeric answer") from exc
    converted = float(number)
    if not math.isfinite(converted) or (number and converted == 0):
        raise ValueError("numeric answer is outside the supported range")
    return text, number


def numeric_claim(claim: str) -> Decimal:
    """Read a bare claim or an explicit terminal answer after a date-qualified description."""
    try:
        return numeric_answer(claim)[1]
    except ValueError:
        match = TERMINAL_ANSWER.search(claim)
        if match and not re.search(r"[0-9]", YEAR_QUALIFIER.sub("", claim[:match.start()])):
            return numeric_answer(match[1])[1]
        raise ValueError("claim needs one explicit numeric answer") from None


def typed_answer(value, answer_type: str) -> tuple[Answer, str]:
    """Validate the task's answer type and derive the display claim from that one value."""
    if isinstance(value, bool):
        raise ValueError("output cannot be a JSON boolean")
    if answer_type in NUMERIC_TYPES:
        text, _ = numeric_answer(value)
        return (text if isinstance(value, str) else value), text
    if answer_type == "string_exact" and type(value) is str and value.strip():
        return value.strip(), value.strip()
    if answer_type == "list_exact" and isinstance(value, list):
        items = []
        for item in value:
            if type(item) is str and item.strip():
                items.append(item.strip())
            elif type(item) in {int, float}:
                numeric_answer(item)
                items.append(item)
            else:
                raise ValueError("list answers must contain only finite numbers or nonempty strings")
        return items, json.dumps(items, ensure_ascii=False)
    raise ValueError(f"output must be {ANSWER_DESCRIPTIONS[answer_type]}")


def benchmark_claim_key(claim: str, answer_type: str = "numeric_exact"):
    try:
        if answer_type in NUMERIC_TYPES:
            return numeric_claim(claim)
        if answer_type == "list_exact":
            values, _ = typed_answer(json.loads(claim), answer_type)
            return tuple(normalize_claim(value) if isinstance(value, str) else numeric_answer(value)[1]
                         for value in values)
    except ValueError:
        pass
    return normalize_claim(claim)


class ToolAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["tool"]
    name: Literal["list_files", "inspect", "query", "calculate"]
    arguments: dict = Field(default_factory=dict)


class DiscussionAction(AgentDecision):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["contribution", "final"]
    output: Answer | None = None

    @model_validator(mode="after")
    def final_has_answer(self):
        if self.kind == "final":
            if self.tag not in {PXPTag.REVISE, PXPTag.RATIFY}:
                raise ValueError("a final submission must REVISE or RATIFY a claim")
            if self.output is None:
                raise ValueError("a final submission requires output")
        elif self.output is not None:
            raise ValueError("only kind=final may include an output")
        return self


ACTION = TypeAdapter(Annotated[Union[ToolAction, DiscussionAction], Field(discriminator="kind")])


class LegacyOpeningSubmission(DiscussionAction):
    kind: Literal["final"]
    tag: Literal[PXPTag.REVISE]
    target_contribution_id: None = None
    output: Answer


class AnswerRevision(BaseModel):
    """Model-facing action. The board claim is derived from the single answer field."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["final"]
    tag: Literal[PXPTag.REVISE]
    target_contribution_id: str | None = None
    output: Answer
    confidence: float = Field(ge=0, le=1)
    explanation: DecisionExplanation

    def to_discussion(self, answer_type: str) -> DiscussionAction:
        output, claim = typed_answer(self.output, answer_type)
        return DiscussionAction(
            kind=self.kind, tag=self.tag, target_contribution_id=self.target_contribution_id,
            output=output, explanation=self.explanation,
            prediction=DecisionPrediction(claim=claim, confidence=self.confidence),
        )


class OpeningSubmission(AnswerRevision):
    target_contribution_id: None = None


class RootRatification(BaseModel):
    """An explicit vote for the current root and the reasoning attached to it."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["ratify"]
    target_contribution_id: str
    confidence: float = Field(ge=0, le=1)
    explanation: DecisionExplanation


class Challenge(DiscussionAction):
    kind: Literal["contribution"]
    tag: Literal[PXPTag.REFUTE, PXPTag.REJECT]
    target_contribution_id: str
    output: None = None


OPENING_ACTION = TypeAdapter(Annotated[Union[ToolAction, OpeningSubmission], Field(discriminator="kind")])
REVIEW_ACTION = TypeAdapter(Annotated[
    Union[ToolAction, AnswerRevision, RootRatification, Challenge], Field(discriminator="kind")
])
CLOSED_REVIEW_ACTION = TypeAdapter(Annotated[
    Union[AnswerRevision, RootRatification, Challenge], Field(discriminator="kind")
])


def active_root(snapshot):
    return next((c for c in reversed(snapshot.contributions) if c.tag == PXPTag.REVISE), None)


def root_id(snapshot, contribution_id):
    by_id = {c.contribution_id: c for c in snapshot.contributions}
    seen = set()
    while contribution_id in by_id and contribution_id not in seen:
        seen.add(contribution_id)
        contribution = by_id[contribution_id]
        if contribution.tag == PXPTag.REVISE:
            return contribution_id
        contribution_id = contribution.target_contribution_id
    return None


ACTION_HELP = """Return one JSON object matching the provided schema.
If the current claim and reasoning are supported, use kind=ratify and its exact target_contribution_id.
If data support a correction to the answer or reasoning, use kind=final and tag=REVISE. Put the answer in output.
Put units in the explanation. A revision resets all votes.
If you find a problem but cannot yet propose a correction, use kind=contribution and tag=REFUTE or REJECT.
Explain the problem and target the current claim. All three agents, including its author, must vote after a revision.
Requesting a tool is not a PXP vote. The controller returns its observation before your next action.
Never claim success from consensus; grading occurs independently after the session stops.
"""


class BenchmarkAgent(PEXAgent):
    """Uses the ordinary broker, prompt budget, parse repairs, personas, and usage ledger."""

    _phase = "review"
    _review_root_id = None

    def __init__(self, *args, answer_type="numeric_exact", **kwargs):
        super().__init__(*args, **kwargs)
        if answer_type not in ANSWER_DESCRIPTIONS:
            raise ValueError(f"Unsupported benchmark answer type: {answer_type}")
        self.answer_type = answer_type

    def claim_key(self, claim):
        return benchmark_claim_key(claim, self.answer_type)

    def output_schema(self):
        if self._phase == "collect":
            return ToolAction.model_json_schema()
        if self._phase == "opening":
            schema = OPENING_ACTION.json_schema()
            revision = schema["$defs"]["OpeningSubmission"]
        elif self._phase == "submit":
            schema = revision = OpeningSubmission.model_json_schema()
        else:
            action = REVIEW_ACTION if self._phase == "review" else CLOSED_REVIEW_ACTION
            schema = action.json_schema()
            revision = schema["$defs"]["AnswerRevision"]
        if self.answer_type in NUMERIC_TYPES:
            answer_schema = {"type": "number"}
        elif self.answer_type == "string_exact":
            answer_schema = {"type": "string", "minLength": 1}
        else:
            answer_schema = {"type": "array", "items": {"anyOf": [{"type": "number"}, {"type": "string"}]}}
        revision["properties"]["output"] = answer_schema
        if self._review_root_id and self._phase in {"review", "review_closed"}:
            target = {"const": self._review_root_id, "type": "string"}
            schema["$defs"]["RootRatification"]["properties"]["target_contribution_id"] = target
            schema["$defs"]["Challenge"]["properties"]["target_contribution_id"] = target.copy()
        return schema

    def parse_output(self, text):
        try:
            raw = json.loads(text)
            json.dumps(raw, allow_nan=False)
            if isinstance(raw, dict) and raw.get("kind") == "ratify":
                return RootRatification.model_validate(raw)
            if isinstance(raw, dict) and raw.get("kind") == "final":
                value, claim = typed_answer(raw.get("output"), self.answer_type)
                raw["output"] = value
                opening = self._phase in {"opening", "submit"}
                if "prediction" not in raw:
                    action = OpeningSubmission if opening else AnswerRevision
                    return action.model_validate(raw).to_discussion(self.answer_type)
                action = LegacyOpeningSubmission if opening else DiscussionAction
                discussion = action.model_validate(raw)
                if self.claim_key(claim) != self.claim_key(discussion.prediction.claim):
                    raise ValueError("final output must equal prediction.claim")
                return discussion
            return ACTION.validate_python(raw)
        except (ValueError, ValidationError) as exc:
            raise DecisionParseError(str(exc)) from exc

    def validate_output(self, decision, snapshot):
        if isinstance(decision, ToolAction):
            if self._phase not in {"collect", "opening", "review"}:
                raise DecisionParseError("Tools are closed for this turn. Return a final submission or a PXP contribution.")
            args = decision.arguments
            if decision.name == "query" and "header" not in args:
                args = {**args, "header": 100}  # reserve space for the largest inherited header
            if len(json.dumps([decision.name, args], sort_keys=True)) > 8000:
                raise DecisionParseError("tool arguments exceed 8000 characters")
            return decision, []
        if self._phase == "collect":
            raise DecisionSemanticError("evidence_required", "Request a CSV tool. The board opens after a successful query with data.")
        if self._phase in {"opening", "submit"} and (
            not isinstance(decision, DiscussionAction) or decision.kind != "final" or decision.tag != PXPTag.REVISE
        ):
            raise DecisionSemanticError("submission_required", "Submit the evidence-based answer with kind=final, tag=REVISE and output.")
        if isinstance(decision, RootRatification):
            root = active_root(snapshot)
            if root is None or decision.target_contribution_id != root.contribution_id:
                raise DecisionSemanticError("stale_vote", "Use kind=ratify with the exact current claim root ID.")
            decision = DiscussionAction(
                kind="contribution", tag=PXPTag.RATIFY, target_contribution_id=decision.target_contribution_id,
                prediction=DecisionPrediction(claim=root.payload.prediction.claim, confidence=decision.confidence),
                explanation=decision.explanation,
            )
        validated, repairs = validate_decision(decision, snapshot, claim_key=self.claim_key)
        if validated.tag == PXPTag.RATIFY:
            root = active_root(snapshot)
            target = next((c for c in snapshot.contributions
                           if c.contribution_id == validated.target_contribution_id), None)
            if (root is None or target is None or target.tag not in {PXPTag.REVISE, PXPTag.RATIFY}
                    or root_id(snapshot, target.contribution_id) != root.contribution_id):
                raise DecisionSemanticError("stale_vote", "RATIFY must target the current REVISE claim or one of its ratifications")
            if self.claim_key(validated.prediction.claim) != self.claim_key(root.payload.prediction.claim):
                raise DecisionSemanticError("different_claim", "RATIFY must repeat the current root claim")
        return validated, repairs

    def read_with_report(self, snapshot, extra_instructions=None, must_keep_ids=(), *, reserved_tokens=0):
        root = active_root(snapshot)
        if self._phase == "collect":
            instructions = (
                "Collect evidence before proposing an answer. Return one tool request as JSON. "
                "Use inspect to locate columns if needed, then query the CSV for the requested rows or aggregate. "
                "Match every qualifier in the question, including date range, category, and source when present. "
                "Use previous observations to choose different arguments.\n" + TOOL_HELP
            )
        elif self._phase in {"opening", "submit"}:
            instructions = (
                "Check whether the observed query and its result cover every qualifier in the question. "
                "A successful query can still be too broad. "
                "For submission use kind=final, tag=REVISE, and target_contribution_id=null. "
                "Put the answer in output. Cite the rows or calculation in your explanation."
            )
            if self._phase == "opening":
                instructions += (
                    "\nIf the existing evidence answers the question, submit it now. "
                    "If a qualifier is missing, request one new, more specific query or calculation.\n" + TOOL_HELP
                )
            else:
                instructions += "\nTools are closed for this turn. Submit the best answer supported by the observed CSV data."
        else:
            instructions = ACTION_HELP
            if self._phase == "review":
                instructions += "\nUse tools when you need to check the claim. Reuse saved evidence.\n" + TOOL_HELP
            else:
                instructions += "\nTools are closed for this turn. Review the current claim using saved evidence."
        instructions += f"\nThe final output must be {ANSWER_DESCRIPTIONS[self.answer_type]}."
        if root:
            instructions += (
                f"\nCurrent claim root id={root.contribution_id}. "
                f'For kind=ratify, REFUTE, or REJECT on this claim, set "target_contribution_id":"{root.contribution_id}".'
            )
            must_keep_ids = tuple(set(must_keep_ids) | {root.contribution_id})
        if extra_instructions:
            instructions += "\n" + extra_instructions
        budget = self.prompt_token_budget
        return build_prompt(
            self.persona, snapshot, self.agent_id, self.system_addendum, self.max_history,
            token_budget=None if budget is None else max(budget - reserved_tokens, 0), must_keep_ids=must_keep_ids,
            output_contract=instructions, next_action="Return the JSON action required by the current phase.",
            empty_board="(no contributions yet)",
        )

    def action(self, snapshot, extra_instructions=None, *, phase="review"):
        previous = self._phase, self._review_root_id
        self._phase = phase
        root = active_root(snapshot)
        self._review_root_id = root.contribution_id if root else None
        try:
            formulation = self.formulate(snapshot, extra_instructions)
        finally:
            self._phase, self._review_root_id = previous
        decision = formulation.decision
        if isinstance(decision, ToolAction):
            return decision
        return self.to_contribution(
            snapshot, decision, tokens=formulation.tokens, latency_ms=formulation.latency_ms,
            final_submission=decision.kind == "final", final_output=decision.output,
            parse_attempts=formulation.attempts, repair_reasons=formulation.repair_reasons,
            raw_responses=formulation.raw_responses, normalizations=formulation.normalizations,
            prompt_budget=formulation.prompt_budget.model_dump() if formulation.prompt_budget else None,
        )
