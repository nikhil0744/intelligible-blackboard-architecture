"""Build an agent prompt inside a token budget and record what was left out (task S2.5).

Order of preservation when the board does not fit:
1. the system prompt and the task description (never truncated);
2. the task context (shortened only as a last resort, and recorded);
3. protected posts: the active position, the post the agent was asked to respond to,
   and the posts those depend on (their targets);
4. the remaining posts, newest first, until the budget is used up.

Token counts here are estimates (no tokenizer is loaded in-process). The estimate is
deliberately pessimistic, and each call's estimate is stored next to the backend's real
count in the usage ledger, so the gap can be checked after a pilot.
"""

from __future__ import annotations

import json
from typing import Callable, List, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, Field

from contracts.schemas import AgentContribution, BlackboardSnapshot, PXPTag
from llm_broker.types import ChatMessage

from .personas import Persona
from .templates import _short, render_contribution, system_prompt

TokenCounter = Callable[[str], int]

PER_MESSAGE_OVERHEAD = 8  # chat-template tokens around each message
MAX_DEPENDENCY_DEPTH = 3


def estimate_tokens(text: str) -> int:
    """Pessimistic estimate: ~3 characters per token for Latin text, 1 token per CJK/wide character."""
    wide = sum(1 for ch in text if ord(ch) >= 0x2E80)
    return wide + -(-(len(text) - wide) // 3)


class PromptBudgetReport(BaseModel):
    """What the prompt builder did. Stored in the contribution metadata."""

    token_budget: Optional[int] = None  # None = no budget enforced
    estimated_prompt_tokens: int = 0
    posts_on_board: int = 0
    posts_included: int = 0
    omitted_contribution_ids: List[str] = Field(default_factory=list)
    protected_contribution_ids: List[str] = Field(default_factory=list)
    context_chars_total: int = 0
    context_chars_kept: int = 0
    over_budget: bool = False  # True if even the protected material did not fit

    @property
    def truncated(self) -> bool:
        return bool(self.omitted_contribution_ids) or self.context_chars_kept < self.context_chars_total


def active_position(snapshot: BlackboardSnapshot) -> Optional[AgentContribution]:
    """The post that currently carries the leading claim: the latest post stating it, else the latest REVISE."""
    if snapshot.active_claim is not None:
        for c in reversed(snapshot.contributions):
            if c.payload.prediction.claim == snapshot.active_claim.claim:
                return c
    for c in reversed(snapshot.contributions):
        if c.tag == PXPTag.REVISE:
            return c
    return None


def protected_ids(snapshot: BlackboardSnapshot, must_keep_ids: Sequence[str] = ()) -> List[str]:
    by_id = {c.contribution_id: c for c in snapshot.contributions}
    roots = [i for i in must_keep_ids if i in by_id]
    ap = active_position(snapshot)
    if ap is not None:
        roots.append(ap.contribution_id)
    keep: List[str] = []
    for root in roots:
        cur, depth = by_id.get(root), 0
        while cur is not None and depth <= MAX_DEPENDENCY_DEPTH and cur.contribution_id not in keep:
            keep.append(cur.contribution_id)
            cur, depth = by_id.get(cur.target_contribution_id or ""), depth + 1
    return keep


def build_prompt(
    persona: Persona,
    snapshot: BlackboardSnapshot,
    agent_id: str,
    extra_instructions: Optional[str] = None,
    max_history: int = 12,
    token_budget: Optional[int] = None,
    must_keep_ids: Sequence[str] = (),
    count_tokens: TokenCounter = estimate_tokens,
    max_context_chars: int = 6000,
) -> Tuple[List[ChatMessage], PromptBudgetReport]:
    """Return (messages, report). With `token_budget=None` this is the unbudgeted prompt plus an estimate."""
    system = system_prompt(persona, extra_instructions)
    contribs = snapshot.contributions
    protected = protected_ids(snapshot, must_keep_ids)
    report = PromptBudgetReport(
        token_budget=token_budget,
        posts_on_board=len(contribs),
        protected_contribution_ids=protected,
    )

    ctx_full = ""
    if snapshot.initial_context:
        ctx_full = json.dumps(snapshot.initial_context, ensure_ascii=False, default=str)
    report.context_chars_total = len(ctx_full)
    ctx = _short(ctx_full, max_context_chars) if ctx_full else ""

    closing = f"\n\nYou are agent '{agent_id}'. Read the board and post your next contribution as JSON."

    def head(context: str) -> List[str]:
        lines = [f"TASK ({snapshot.task_id}):", snapshot.task_description.strip()]
        if context:
            lines += ["", "CONTEXT:", context]
        if snapshot.active_claim:
            lines += ["", f"CURRENT LEADING CLAIM: {snapshot.active_claim.claim}"]
        lines += ["", f"BLACKBOARD (status={snapshot.status.value}, {len(contribs)} posts):"]
        return lines

    def cost(text: str) -> int:
        return count_tokens(text) + 1  # + newline

    fixed = count_tokens(system) + 2 * PER_MESSAGE_OVERHEAD + count_tokens(closing) + 16  # 16: omitted-posts line

    rendered = {c.contribution_id: render_contribution(c) for c in contribs}
    prot: Set[str] = set(protected)
    protected_cost = sum(cost(rendered[i]) for i in prot)

    # 2. shorten the context only if the fixed part + protected posts cannot fit otherwise
    if token_budget is not None and ctx:
        room = token_budget - fixed - protected_cost - sum(cost(x) for x in head(""))
        while ctx and cost(ctx) > max(room, 0):
            ctx = _short(ctx, max(len(ctx) * 3 // 4, 0)) if len(ctx) > 40 else ""
    report.context_chars_kept = len(ctx)
    head_lines = head(ctx)
    used = fixed + sum(cost(x) for x in head_lines) + protected_cost

    # 3 + 4. protected posts always; then the most recent others while they fit
    window = contribs[-max_history:] if max_history else []
    include: Set[str] = set(prot)
    for c in reversed(window):
        if c.contribution_id in include:
            continue
        need = cost(rendered[c.contribution_id])
        if token_budget is not None and used + need > token_budget:
            break
        include.add(c.contribution_id)
        used += need

    kept = [c for c in contribs if c.contribution_id in include]
    report.posts_included = len(kept)
    report.omitted_contribution_ids = [c.contribution_id for c in contribs if c.contribution_id not in include]
    report.estimated_prompt_tokens = used
    report.over_budget = token_budget is not None and used > token_budget

    lines = list(head_lines)
    if report.omitted_contribution_ids:
        lines.append(f"(... {len(report.omitted_contribution_ids)} other posts omitted ...)")
    lines += [rendered[c.contribution_id] for c in kept] or [
        "(empty - you are posting the first hypothesis; use REVISE with target null)"
    ]
    user = "\n".join(lines) + closing
    return [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)], report
