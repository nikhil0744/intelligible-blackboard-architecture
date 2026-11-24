"""Benchmark actions extend PEXAgent's validated inference path, not a second agent loop."""
from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

from agents.base import PEXAgent
from contracts.schemas import PXPTag
from prompts import AgentDecision, DecisionParseError, DecisionSemanticError
from prompts.validation import normalize_claim

from .data_tools import TOOL_HELP


class ToolAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["tool"]
    name: Literal["list_files", "inspect", "query", "calculate"]
    arguments: dict = Field(default_factory=dict)


class DiscussionAction(AgentDecision):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["contribution", "final"]
    output: int | float | str | None = None

    @model_validator(mode="after")
    def final_is_numeric(self):
        if self.kind == "final":
            if self.tag not in {PXPTag.REVISE, PXPTag.RATIFY}:
                raise ValueError("a final submission must REVISE or RATIFY a claim")
            try:
                value = Decimal(str(self.output))
                claim = Decimal(self.prediction.claim.strip())
            except (ValueError, InvalidOperation):
                raise ValueError("this profile requires a numeric output and the same numeric claim")
            if not value.is_finite() or not claim.is_finite() or value != claim:
                raise ValueError("final output must equal the numeric prediction.claim")
        elif self.output is not None:
            raise ValueError("only kind=final may include an output")
        return self


ACTION = TypeAdapter(Annotated[Union[ToolAction, DiscussionAction], Field(discriminator="kind")])


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


ACTION_HELP = """For this benchmark, return ONE JSON action (this replaces the earlier output shape):
Discussion: {"kind":"contribution","tag":"REVISE|RATIFY|REFUTE|REJECT","target_contribution_id":null,
   "prediction":{"claim":"short claim","confidence":0.5},"explanation":{"rationale":"why","evidence":[]}}
Final submission: same as discussion, but "kind":"final", plus "output":123.
For final submissions, prediction.claim must be the same bare number as output; include units in the explanation.
Once the observations support a numeric answer, submit it as "kind":"final" using REVISE or RATIFY.
A REVISE creates a fresh claim and resets all votes. RATIFY the current root id and repeat its claim exactly;
all THREE agents, including the author, must explicitly RATIFY after the latest revision to reach consensus.
Requesting a tool is not a PXP vote. The controller returns its observation before your next action.
Never claim success from consensus; grading occurs independently after the session stops.
"""


class BenchmarkAgent(PEXAgent):
    """Uses the ordinary broker, prompt budget, parse repairs, personas, and usage ledger."""

    _tools_allowed = True

    def output_schema(self):
        return ACTION.json_schema() if self._tools_allowed else DiscussionAction.model_json_schema()

    def parse_output(self, text):
        try:
            raw = json.loads(text)
            # bool is a numeric subclass in Python; never turn false into a numeric answer.
            if isinstance(raw, dict) and isinstance(raw.get("output"), bool):
                raise ValueError("numeric final output cannot be a boolean")
            return ACTION.validate_python(raw)
        except (ValueError, ValidationError) as exc:
            raise DecisionParseError(str(exc)) from exc

    def validate_output(self, decision, snapshot):
        if isinstance(decision, ToolAction):
            if not self._tools_allowed:
                raise DecisionParseError("Tools are closed for this turn. Return a final submission or a PXP contribution.")
            if len(json.dumps(decision.arguments)) > 8000:
                raise DecisionParseError("tool arguments exceed 8000 characters")
            return decision, []
        validated, repairs = super().validate_output(decision, snapshot)
        if validated.tag == PXPTag.RATIFY:
            root = active_root(snapshot)
            if root is None or root_id(snapshot, validated.target_contribution_id) != root.contribution_id:
                raise DecisionSemanticError("stale_vote", "RATIFY must target the current REVISE claim or one of its ratifications")
            if normalize_claim(validated.prediction.claim) != normalize_claim(root.payload.prediction.claim):
                raise DecisionSemanticError("different_claim", "RATIFY must repeat the current root claim")
        return validated, repairs

    def read_with_report(self, snapshot, extra_instructions=None, must_keep_ids=()):
        root = active_root(snapshot)
        instructions = ACTION_HELP
        if self._tools_allowed:
            instructions += (
                '\nTool request shape: {"kind":"tool","name":"inspect","arguments":{"file":"a permitted CSV"}}. '
                "Use the tools when you need evidence. Reuse prior observations and avoid repeating requests."
                "\n" + TOOL_HELP
            )
        else:
            instructions += "\nTools are closed for this turn. Post a final answer if supported, or a PXP contribution explaining the uncertainty."
        if root:
            instructions += f"\nCurrent claim root id={root.contribution_id}."
            must_keep_ids = tuple(set(must_keep_ids) | {root.contribution_id})
        if extra_instructions:
            instructions += "\n" + extra_instructions
        return super().read_with_report(snapshot, instructions, must_keep_ids)

    def action(self, snapshot, extra_instructions=None, *, tools_allowed=True):
        previous = self._tools_allowed
        self._tools_allowed = tools_allowed
        try:
            formulation = self.formulate(snapshot, extra_instructions)
        finally:
            self._tools_allowed = previous
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
