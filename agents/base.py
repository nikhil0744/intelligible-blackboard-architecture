"""Base PEX agent: read board -> formulate prediction + explanation -> select PXP tag -> submit."""

from __future__ import annotations

import logging
import time
from typing import List, Optional

from contracts.schemas import (
    AgentContribution,
    AgentRole,
    BlackboardSnapshot,
    PXPTag,
)
from llm_broker import ChatMessage, LLMError, LLMRequest, ModelBroker
from prompts import (
    AgentDecision,
    DecisionParseError,
    Persona,
    build_messages,
    decision_json_schema,
    get_persona,
    parse_decision,
    repair_message,
)

from .board_client import BoardClient

log = logging.getLogger(__name__)


class AgentTurnError(RuntimeError):
    pass


class PEXAgent:
    """A prediction+explanation agent bound to one persona and one broker.

    Stateless between turns: everything it knows comes from the snapshot, so S3 can
    replay it on a deep-copied/modified history (`act(..., is_counterfactual=True)`).
    `system_addendum` is the hook S3's recovery loop uses to adjust the agent's prompt.
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
        self.counterfactual_capable = counterfactual_capable
        self.system_addendum: Optional[str] = None

    @property
    def role(self) -> AgentRole:
        return AgentRole.COUNTERFACTUAL if self.counterfactual_capable else self.persona.role

    # ---- 1. read ------------------------------------------------------------
    def read(self, snapshot: BlackboardSnapshot, extra_instructions: Optional[str] = None) -> List[ChatMessage]:
        extra = "\n".join(x for x in (self.system_addendum, extra_instructions) if x) or None
        return build_messages(self.persona, snapshot, self.agent_id, extra, self.max_history)

    # ---- 2. formulate ----------------------------------------------------------
    def formulate(self, snapshot: BlackboardSnapshot, extra_instructions: Optional[str] = None):
        """Returns (decision, tokens_used, latency_ms, attempts). Re-prompts on invalid JSON."""
        messages = self.read(snapshot, extra_instructions)
        tokens, latency, last_err = 0, 0.0, ""
        for attempt in range(1, self.max_parse_retries + 2):
            req = LLMRequest(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=self.seed,
                json_schema=decision_json_schema() if self.use_json_schema else None,
                agent_id=self.agent_id,
            )
            try:
                resp = self.broker.generate(req)
            except LLMError as e:
                raise AgentTurnError(f"{self.agent_id}: model call failed: {e}") from e
            tokens += resp.total_tokens
            latency += resp.latency_ms
            try:
                return parse_decision(resp.text), tokens, latency, attempt
            except DecisionParseError as e:
                last_err = str(e)
                log.warning("%s: invalid PEX JSON (attempt %d): %s", self.agent_id, attempt, last_err[:200])
                messages = messages + [
                    ChatMessage(role="assistant", content=resp.text[:2000]),
                    repair_message(resp.text, last_err),
                ]
        raise AgentTurnError(f"{self.agent_id}: no valid PEX JSON after retries: {last_err[:300]}")

    # ---- 3. select tag (normalise against PXP rules) ---------------------------
    def select_tag(self, decision: AgentDecision, snapshot: BlackboardSnapshot) -> AgentDecision:
        contribs = snapshot.contributions
        ids = {c.contribution_id for c in contribs}
        d = decision.model_copy(deep=True)
        if not contribs:
            # empty board: the only meaningful move is an opening hypothesis
            d.tag, d.target_contribution_id = PXPTag.REVISE, None
            return d
        if d.target_contribution_id not in ids:
            # small models often truncate UUIDs: accept a unique prefix, else fall back
            t = (d.target_contribution_id or "").strip()
            hits = [i for i in ids if len(t) >= 6 and i.startswith(t)]
            d.target_contribution_id = hits[0] if len(hits) == 1 else self._default_target(snapshot)
        # RATIFY means agreeing with the target: inherit its claim if the model drifted
        if d.tag == PXPTag.RATIFY:
            target = next(c for c in contribs if c.contribution_id == d.target_contribution_id)
            if d.prediction.claim.strip().lower() != target.payload.prediction.claim.strip().lower():
                d.prediction.summary = d.prediction.summary or d.prediction.claim
                d.prediction.claim = target.payload.prediction.claim
        return d

    def _default_target(self, snapshot: BlackboardSnapshot) -> str:
        """Latest post by another agent, else the latest post."""
        for c in reversed(snapshot.contributions):
            if c.agent_id != self.agent_id:
                return c.contribution_id
        return snapshot.contributions[-1].contribution_id

    # ---- build contract object -------------------------------------------------
    def to_contribution(
        self,
        snapshot: BlackboardSnapshot,
        decision: AgentDecision,
        tokens: int = 0,
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
                **metadata,
            },
        )

    # ---- full lifecycle ----------------------------------------------------------
    def act(
        self,
        snapshot: BlackboardSnapshot,
        extra_instructions: Optional[str] = None,
        is_counterfactual: bool = False,
    ) -> AgentContribution:
        """read -> formulate -> select tag -> AgentContribution (does NOT write to the board)."""
        t0 = time.perf_counter()
        decision, tokens, _, attempts = self.formulate(snapshot, extra_instructions)
        decision = self.select_tag(decision, snapshot)
        return self.to_contribution(
            snapshot,
            decision,
            tokens=tokens,
            latency_ms=(time.perf_counter() - t0) * 1000,
            is_counterfactual=is_counterfactual,
            parse_attempts=attempts,
        )

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
