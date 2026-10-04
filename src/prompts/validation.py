"""Semantic validation of a parsed AgentDecision against the board (task S2.2).

`parse_decision` only checks the JSON shape. This module checks that the decision
makes sense on the current board, and it never rewrites the meaning of what the
model said: a RATIFY whose claim differs from its target, or a target that is not
on the board, is an error to re-prompt, not something to patch into agreement.

Student 3's replay must call `validate_decision` too, so that ordinary turns and
replayed turns accept exactly the same outputs.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from contracts.schemas import BlackboardSnapshot, PXPTag

from .parser import DecisionParseError
from .schema import AgentDecision

MIN_ID_PREFIX = 6

# Repair reason codes (recorded in contribution metadata / failure records).
REPAIR_DROPPED_TARGET_EMPTY_BOARD = "dropped_target_on_empty_board"
REPAIR_COMPLETED_ID_PREFIX = "completed_unique_id_prefix"


class DecisionSemanticError(DecisionParseError):
    """Well-formed JSON whose content is inconsistent with the board.

    `code` is a stable machine-readable reason; the message is written for the
    model, because it is sent back verbatim in the repair prompt.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def normalize_claim(claim: str) -> str:
    """Case, whitespace and trailing punctuation are not a difference in position."""
    return re.sub(r"\s+", " ", claim).strip().rstrip(".!;,").strip().lower()


def _ids_hint(snapshot: BlackboardSnapshot, limit: int = 12) -> str:
    return ", ".join(c.contribution_id for c in snapshot.contributions[-limit:])


def validate_decision(decision: AgentDecision, snapshot: BlackboardSnapshot) -> Tuple[AgentDecision, List[str]]:
    """Return (decision, repair_reasons) or raise DecisionSemanticError.

    Only two meaning-preserving repairs are applied, and both are reported:
    - a REVISE on an empty board drops a target id that cannot exist;
    - a truncated id is completed when it is a unique prefix of one board id.
    """
    d = decision.model_copy(deep=True)
    repairs: List[str] = []
    contribs = snapshot.contributions

    if not contribs:
        if d.tag != PXPTag.REVISE:
            raise DecisionSemanticError(
                "non_revise_on_empty_board",
                f"The board is empty, so there is nothing to {d.tag.value}. "
                "Post your first hypothesis with tag REVISE and target_contribution_id null.",
            )
        if d.target_contribution_id is not None:
            d.target_contribution_id = None
            repairs.append(REPAIR_DROPPED_TARGET_EMPTY_BOARD)
        return d, repairs

    ids = {c.contribution_id for c in contribs}
    target_id = d.target_contribution_id

    if target_id is None:
        if d.tag != PXPTag.REVISE:
            raise DecisionSemanticError(
                "missing_target",
                f"{d.tag.value} must set target_contribution_id to the id of the post you are responding to. "
                f"Valid ids: {_ids_hint(snapshot)}.",
            )
        return d, repairs

    if target_id not in ids:
        t = target_id.strip()
        hits = [i for i in ids if len(t) >= MIN_ID_PREFIX and i.startswith(t)]
        if len(hits) != 1:
            raise DecisionSemanticError(
                "unknown_target",
                f"target_contribution_id '{target_id}' is not an id on the board. "
                f"Copy one of these ids exactly: {_ids_hint(snapshot)}."
                + (" A REVISE may also use null." if d.tag == PXPTag.REVISE else ""),
            )
        d.target_contribution_id = hits[0]
        repairs.append(REPAIR_COMPLETED_ID_PREFIX)

    if d.tag == PXPTag.RATIFY:
        target = next(c for c in contribs if c.contribution_id == d.target_contribution_id)
        target_claim = target.payload.prediction.claim
        if normalize_claim(d.prediction.claim) != normalize_claim(target_claim):
            raise DecisionSemanticError(
                "ratify_claim_mismatch",
                f"You tagged RATIFY on {target.contribution_id}, but your claim '{d.prediction.claim}' "
                f"is not the claim of that post ('{target_claim}'). "
                "RATIFY means you hold exactly the target's claim: repeat it word for word. "
                "If your claim is different, use REVISE (your own claim) or REFUTE/REJECT instead.",
            )
        d.prediction.claim = target_claim  # identical up to case/whitespace/punctuation
    return d, repairs
