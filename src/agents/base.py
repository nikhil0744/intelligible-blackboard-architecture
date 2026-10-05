"""Base PEX agent: read board -> formulate prediction + explanation -> select PXP tag -> submit."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from contracts.schemas import (
    AgentContribution,
    AgentRole,
    BlackboardSnapshot,
)
from llm_broker import ChatMessage, LLMError, LLMRequest, ModelBroker
from prompts import (
    AgentDecision,
    DecisionParseError,
    DecisionSemanticError,
    Persona,
    PromptBudgetReport,
    build_prompt,
    estimate_tokens,
    decision_json_schema,
    get_persona,
    parse_decision,
    repair_message,
    validate_decision,
)

from .board_client import BoardClient

log = logging.getLogger(__name__)


@dataclass
class Formulation:
    """Outcome of one agent turn's model calls, including every repair attempt."""

    decision: AgentDecision
    tokens: Optional[int]  # None if the backend did not report usage for some call (unknown, not zero)
    latency_ms: float
    attempts: int
    prompt_budget: Optional[PromptBudgetReport] = None
    raw_responses: List[str] = field(default_factory=list)
    # why each earlier attempt was rejected, e.g. "invalid_json: ..." or "ratify_claim_mismatch: ..."
    repair_reasons: List[str] = field(default_factory=list)
    # meaning-preserving fixes applied to the accepted output (see prompts.validation)
    normalizations: List[str] = field(default_factory=list)


class AgentTurnError(RuntimeError):
    """A turn that produced no valid contribution. It is a recorded failure, never a default answer.

    `kind` is "model_failure" (the model call itself failed) or "invalid_output"
    (the model answered, but never with a valid, board-consistent decision).
    """

    def __init__(
        self,
        message: str,
        *,
        agent_id: str = "",
        kind: str = "invalid_output",
        attempts: int = 0,
        tokens: Optional[int] = 0,
        latency_ms: float = 0.0,
        raw_responses: Optional[List[str]] = None,
        repair_reasons: Optional[List[str]] = None,
    ):
        super().__init__(message)
        self.agent_id = agent_id
        self.kind = kind
        self.attempts = attempts
        self.tokens = tokens
        self.latency_ms = latency_ms
        self.raw_responses = list(raw_responses or [])
        self.repair_reasons = list(repair_reasons or [])

    def to_record(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "kind": self.kind,
            "error": str(self),
            "attempts": self.attempts,
            "tokens": self.tokens,
            "latency_ms": self.latency_ms,
            "raw_responses": self.raw_responses,
            "repair_reasons": self.repair_reasons,
        }


class PEXAgent:
    """A prediction+explanation agent bound to one persona and one broker.

    Stateless between turns: everything it knows comes from the snapshot, so S3 can
    replay it on a deep-copied/modified history (`act(..., is_counterfactual=True)`).
    `system_addendum` is the hook S3's recovery loop uses to adjust the agent's prompt.

    Generating is separate from committing: `act()` / calling the agent only returns a
    contribution; the scheduler (S1) decides whether and in what order to commit it.

    `counterfactual_capable` is the experimental variable and nothing else: it does not
    change the agent's role, persona, prompt, model, decoding settings or seed, so the
    0/1/2/3-capable conditions run the same roster.
    """

    def __init__(
        self,
        agent_id: str,
        persona: Persona | str,
        broker: ModelBroker,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 768,
        max_parse_retries: int = 2,
        max_history: int = 12,
        seed: Optional[int] = None,
        use_json_schema: bool = True,
        counterfactual_capable: bool = False,
        context_tokens: Optional[int] = 8192,
        prompt_margin_tokens: int = 64,
        reasoning_first: bool = False,
    ):
        self.agent_id = agent_id
        self.persona = get_persona(persona) if isinstance(persona, str) else persona
        self.broker = broker
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_parse_retries = max_parse_retries
        self.max_history = max_history
        self.seed = seed
        self.use_json_schema = use_json_schema
        self.reasoning_first = reasoning_first
        self.counterfactual_capable = counterfactual_capable
        # Must equal the backend's context window (LLM_NUM_CTX). None disables the prompt budget.
        self.context_tokens = context_tokens
        self.prompt_margin_tokens = prompt_margin_tokens
        self.system_addendum: Optional[str] = None

    @property
    def role(self) -> AgentRole:
        """Domain role of the persona. Counterfactual capability never changes it."""
        return self.persona.role

    def spec(self) -> Dict[str, Any]:
        """Agent specification for run manifests: who this agent is and how it decodes."""
        return {
            "agent_id": self.agent_id,
            "persona": self.persona.id,
            "role": self.role.value,
            "domain": self.persona.domain,
            "counterfactual_capable": self.counterfactual_capable,
            "model": self.model or self.broker.default_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_parse_retries": self.max_parse_retries,
            "max_history": self.max_history,
            "context_tokens": self.context_tokens,
            "prompt_token_budget": self.prompt_token_budget,
            "seed": self.seed,
            "use_json_schema": self.use_json_schema,
            "reasoning_first": self.reasoning_first,
        }

    @property
    def prompt_token_budget(self) -> Optional[int]:
        """Tokens available for the prompt: context window minus the reserved response and a margin."""
        if self.context_tokens is None:
            return None
        return max(self.context_tokens - self.max_tokens - self.prompt_margin_tokens, 0)

    # ---- 1. read ------------------------------------------------------------
    def read(self, snapshot: BlackboardSnapshot, extra_instructions: Optional[str] = None) -> List[ChatMessage]:
        return self.read_with_report(snapshot, extra_instructions)[0]

    def read_with_report(
        self,
        snapshot: BlackboardSnapshot,
        extra_instructions: Optional[str] = None,
        must_keep_ids: Sequence[str] = (),
    ):
        """Build the prompt inside the token budget; returns (messages, PromptBudgetReport)."""
        extra = "\n".join(x for x in (self.system_addendum, extra_instructions) if x) or None
        return build_prompt(
            self.persona,
            snapshot,
            self.agent_id,
            extra,
            self.max_history,
            token_budget=self.prompt_token_budget,
            must_keep_ids=must_keep_ids,
        )

    # ---- 2. formulate + validate ----------------------------------------------
    def formulate(
        self,
        snapshot: BlackboardSnapshot,
        extra_instructions: Optional[str] = None,
        must_keep_ids: Sequence[str] = (),
    ) -> Formulation:
        """Call the model until its output is valid JSON AND consistent with the board.

        Each rejected attempt is re-prompted with the actual error (shape or semantic).
        After `max_parse_retries` repairs the turn fails with AgentTurnError; nothing is
        ever patched into agreement. Raw responses and reasons are kept either way.
        """
        messages, budget = self.read_with_report(snapshot, extra_instructions, must_keep_ids)
        tokens: Optional[int] = 0
        latency = 0.0
        raw: List[str] = []
        reasons: List[str] = []
        max_attempts = self.max_parse_retries + 1
        for attempt in range(1, max_attempts + 1):
            req = LLMRequest(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=self.seed,
                json_schema=decision_json_schema(reasoning_first=self.reasoning_first) if self.use_json_schema else None,
                agent_id=self.agent_id,
                # ledger tags; trial_id / phase come from the caller's usage_scope
                metadata={
                    "session_id": snapshot.session_id,
                    "is_repair": attempt > 1,
                    "estimated_prompt_tokens": sum(estimate_tokens(m.content) + 8 for m in messages),
                },
            )
            try:
                resp = self.broker.generate(req)
            except LLMError as e:
                raise AgentTurnError(
                    f"{self.agent_id}: model call failed: {e}",
                    agent_id=self.agent_id,
                    kind="model_failure",
                    attempts=attempt,
                    tokens=tokens,
                    latency_ms=latency,
                    raw_responses=raw,
                    repair_reasons=reasons,
                ) from e
            tokens = None if tokens is None or resp.total_tokens is None else tokens + resp.total_tokens
            latency += resp.latency_ms
            raw.append(resp.text)
            try:
                decision, normalizations = self.validate_output(parse_decision(resp.text), snapshot)
                return Formulation(decision, tokens, latency, attempt, budget, raw, reasons, normalizations)
            except DecisionParseError as e:
                code = e.code if isinstance(e, DecisionSemanticError) else "invalid_json"
                reasons.append(f"{code}: {e}")
                log.warning("%s: rejected output (attempt %d, %s): %s", self.agent_id, attempt, code, str(e)[:200])
                messages = messages + [
                    ChatMessage(role="assistant", content=resp.text[:2000]),
                    repair_message(resp.text, str(e)),
                ]
        raise AgentTurnError(
            f"{self.agent_id}: no valid decision after {max_attempts} attempts: {reasons[-1][:300]}",
            agent_id=self.agent_id,
            kind="invalid_output",
            attempts=max_attempts,
            tokens=tokens,
            latency_ms=latency,
            raw_responses=raw,
            repair_reasons=reasons,
        )

    # ---- 3. select tag ---------------------------------------------------------
    def validate_output(self, decision: AgentDecision, snapshot: BlackboardSnapshot):
        """Validation hook for runners with additional output requirements."""
        return validate_decision(decision, snapshot)

    def select_tag(self, decision: AgentDecision, snapshot: BlackboardSnapshot) -> AgentDecision:
        """Validate a decision against the board (raises DecisionSemanticError). Kept for callers of the old API."""
        return self.validate_output(decision, snapshot)[0]

    # ---- build contract object -------------------------------------------------
    def to_contribution(
        self,
        snapshot: BlackboardSnapshot,
        decision: AgentDecision,
        tokens: Optional[int] = 0,
        latency_ms: float = 0.0,
        is_counterfactual: bool = False,
        **metadata,
    ) -> AgentContribution:
        return AgentContribution(
            session_id=snapshot.session_id,
            turn_index=len(snapshot.contributions),
            agent_id=self.agent_id,
            agent_role=self.role,
            tag=decision.tag,
            target_contribution_id=decision.target_contribution_id,
            payload=decision.to_pex(),
            is_counterfactual=is_counterfactual,
            token_usage=tokens,
            latency_ms=latency_ms,
            metadata={
                "persona": self.persona.id,
                "domain": self.persona.domain,
                "model": self.model or self.broker.default_model,
                "counterfactual_capable": self.counterfactual_capable,
                **metadata,
            },
        )

    # ---- full lifecycle ----------------------------------------------------------
    def act(
        self,
        snapshot: BlackboardSnapshot,
        extra_instructions: Optional[str] = None,
        is_counterfactual: bool = False,
        must_keep_ids: Sequence[str] = (),
    ) -> AgentContribution:
        """read -> formulate + validate -> AgentContribution (does NOT write to the board).

        `token_usage` on the result is None when the backend did not report usage for
        every call of this turn. Per-call detail is in `broker.ledger`.
        """
        t0 = time.perf_counter()
        f = self.formulate(snapshot, extra_instructions, must_keep_ids)
        return self.to_contribution(
            snapshot,
            f.decision,
            tokens=f.tokens,
            latency_ms=(time.perf_counter() - t0) * 1000,
            is_counterfactual=is_counterfactual,
            parse_attempts=f.attempts,
            repair_reasons=f.repair_reasons,
            normalizations=f.normalizations,
            raw_responses=f.raw_responses,
            prompt_budget=f.prompt_budget.model_dump() if f.prompt_budget else None,
        )

    # ---- scheduler handler adapter (S2.1) --------------------------------------
    def __call__(self, snapshot: BlackboardSnapshot, turn: Any = None) -> AgentContribution:
        """Scheduler handler: `scheduler.register_agent(id, role, turn_handler=agent)`.

        Generates a contribution from the snapshot the scheduler hands over and returns
        it uncommitted. If the scheduled turn names a post that is on the board, the agent
        is told which post it was asked to respond to; it still chooses its own tag/target.
        Raises AgentTurnError (a recorded failure) if no valid contribution is produced.
        """
        hint = None
        target = getattr(turn, "target_contribution_id", None)
        if target and any(c.contribution_id == target for c in snapshot.contributions):
            hint = f"The scheduler asked you to respond to the post with id={target}."
        return self.act(snapshot, extra_instructions=hint, must_keep_ids=[target] if hint else ())

    def step(self, board: BoardClient, session_id: str, extra_instructions: Optional[str] = None) -> AgentContribution:
        """act() against the live board, then submit. Retries once if the board moved on."""
        for _ in range(2):
            snapshot = board.get_snapshot(session_id)
            contribution = self.act(snapshot, extra_instructions)
            try:
                board.submit(contribution)
                return contribution
            except ValueError as e:
                if "turn_index" not in str(e):
                    raise
                log.info("%s: board advanced during inference; re-reading", self.agent_id)
        raise AgentTurnError(f"{self.agent_id}: could not submit (board kept advancing)")

    def __repr__(self) -> str:
        cf = ", counterfactual" if self.counterfactual_capable else ""
        return f"PEXAgent({self.agent_id!r}, persona={self.persona.id!r}{cf})"
