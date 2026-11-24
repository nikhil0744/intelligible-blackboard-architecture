"""Benchmark actions extend PEXAgent's validated inference path, not a second agent loop."""
from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

from agents.base import PEXAgent
from contracts.schemas import PXPTag
from prompts import AgentDecision, DecisionParseError, DecisionSemanticError
from prompts.budget import build_prompt
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


class OpeningSubmission(DiscussionAction):
    kind: Literal["final"]
    tag: Literal[PXPTag.REVISE]
    target_contribution_id: None = None


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
For PXP responses, set target_contribution_id to the current claim's ID for RATIFY, REFUTE, or REJECT.
For final submissions, prediction.claim must be the same bare number as output; include units in the explanation.
Once the observations support a numeric answer, submit it as "kind":"final" using REVISE or RATIFY.
A REVISE creates a fresh claim and resets all votes. RATIFY the current root id and repeat its claim exactly;
all THREE agents, including the author, must explicitly RATIFY after the latest revision to reach consensus.
Requesting a tool is not a PXP vote. The controller returns its observation before your next action.
Never claim success from consensus; grading occurs independently after the session stops.
"""


class BenchmarkAgent(PEXAgent):
    """Uses the ordinary broker, prompt budget, parse repairs, personas, and usage ledger."""

    _phase = "review"

    def output_schema(self):
        if self._phase == "collect":
            return ToolAction.model_json_schema()
        if self._phase == "submit":
            return OpeningSubmission.model_json_schema()
        return ACTION.json_schema() if self._phase == "review" else DiscussionAction.model_json_schema()

    def parse_output(self, text):
        try:
            raw = json.loads(text)
            json.dumps(raw, allow_nan=False)
            # bool is a numeric subclass in Python; never turn false into a numeric answer.
            if isinstance(raw, dict) and isinstance(raw.get("output"), bool):
                raise ValueError("numeric final output cannot be a boolean")
            return ACTION.validate_python(raw)
        except (ValueError, ValidationError) as exc:
            raise DecisionParseError(str(exc)) from exc

    def validate_output(self, decision, snapshot):
        if isinstance(decision, ToolAction):
            if self._phase not in {"collect", "review"}:
                raise DecisionParseError("Tools are closed for this turn. Return a final submission or a PXP contribution.")
            if len(json.dumps([decision.name, decision.arguments], sort_keys=True)) > 8000:
                raise DecisionParseError("tool arguments exceed 8000 characters")
            return decision, []
        if self._phase == "collect":
            raise DecisionSemanticError("evidence_required", "Request a CSV tool. The board opens after a successful query with data.")
        if self._phase == "submit" and (decision.kind != "final" or decision.tag != PXPTag.REVISE):
            raise DecisionSemanticError("submission_required", "Submit the evidence-based numeric answer with kind=final and tag=REVISE.")
        validated, repairs = super().validate_output(decision, snapshot)
        if validated.tag == PXPTag.RATIFY:
            root = active_root(snapshot)
            target = next((c for c in snapshot.contributions
                           if c.contribution_id == validated.target_contribution_id), None)
            if (root is None or target is None or target.tag not in {PXPTag.REVISE, PXPTag.RATIFY}
                    or root_id(snapshot, target.contribution_id) != root.contribution_id):
                raise DecisionSemanticError("stale_vote", "RATIFY must target the current REVISE claim or one of its ratifications")
            if normalize_claim(validated.prediction.claim) != normalize_claim(root.payload.prediction.claim):
                raise DecisionSemanticError("different_claim", "RATIFY must repeat the current root claim")
        return validated, repairs

    def read_with_report(self, snapshot, extra_instructions=None, must_keep_ids=()):
        root = active_root(snapshot)
        if self._phase == "collect":
            instructions = (
                "Collect evidence before proposing an answer. Return one tool request as JSON. "
                "For example, list available files with kind=tool, name=list_files, and arguments={}. "
                "Use inspect to locate columns if needed, then query the CSV for the requested rows or aggregate. "
                "Use previous observations to choose different arguments.\n" + TOOL_HELP
            )
        elif self._phase == "submit":
            instructions = (
                "Tools are closed for this turn. Submit a numeric answer from the observed CSV data. "
                "Return kind=final, tag=REVISE, and target_contribution_id=null. "
                "Use the same bare number for prediction.claim and output. Cite the rows or calculation in your explanation."
            )
        else:
            instructions = ACTION_HELP
            if self._phase == "review":
                instructions += "\nUse tools when you need to check the claim. Reuse saved evidence.\n" + TOOL_HELP
            else:
                instructions += "\nTools are closed for this turn. Review the current claim using saved evidence."
        if root:
            instructions += f"\nCurrent claim root id={root.contribution_id}."
            must_keep_ids = tuple(set(must_keep_ids) | {root.contribution_id})
        if extra_instructions:
            instructions += "\n" + extra_instructions
        return build_prompt(
            self.persona, snapshot, self.agent_id, self.system_addendum, self.max_history,
            token_budget=self.prompt_token_budget, must_keep_ids=must_keep_ids,
            output_contract=instructions, next_action="Return the JSON action required by the current phase.",
            empty_board="(no contributions yet)",
        )

    def action(self, snapshot, extra_instructions=None, *, phase="review"):
        previous = self._phase
        self._phase = phase
        try:
            formulation = self.formulate(snapshot, extra_instructions)
        finally:
            self._phase = previous
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
