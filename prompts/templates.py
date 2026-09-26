"""Prompt templates: persona system prompt + rendered blackboard view + JSON output contract."""

from __future__ import annotations

import json
from typing import List, Optional

from contracts.schemas import AgentContribution, BlackboardSnapshot
from llm_broker.types import ChatMessage

from .personas import Persona

PXP_RULES = """You are one agent on a shared BLACKBOARD. Agents never message each other; they post
contributions, each = PREDICTION (what) + EXPLANATION (why) + one PXP TAG:
- RATIFY: you agree with BOTH the target's prediction and its explanation. Restate that prediction.
- REVISE: you propose a new or amended prediction/explanation (also used for the FIRST hypothesis on an empty board).
- REFUTE: you dispute the target's prediction or reasoning with specific counter-evidence, but cannot yet offer a better answer.
- REJECT: you fundamentally disagree with both the target's prediction and explanation.
Rules:
1. RATIFY/REFUTE/REJECT must set target_contribution_id to an id shown on the board. REVISE may target the post it amends, or null.
2. Do not RATIFY just to be agreeable; do not REJECT without concrete reasons.
3. If others' arguments convinced you, REVISE (update your view) instead of repeating yourself.
4. Keep the claim short and final-answer-like; put reasoning in the rationale; list concrete evidence.
5. confidence is a number in [0,1]."""

OUTPUT_CONTRACT = """Respond with ONLY one JSON object, no prose, no markdown fences:
{"tag": "RATIFY|REVISE|REFUTE|REJECT",
 "target_contribution_id": "<id from the board or null>",
 "prediction": {"claim": "...", "confidence": 0.0, "summary": "..."},
 "explanation": {"rationale": "...", "evidence": ["..."], "assumptions": ["..."], "limitations": "..."}}"""


def system_prompt(persona: Persona, extra_instructions: Optional[str] = None) -> str:
    parts = [
        f"ROLE: {persona.title} ({persona.role.value.lower()}, domain: {persona.domain}).",
        f"EXPERTISE: {persona.expertise}",
        f"STANCE: {persona.stance}",
        "",
        PXP_RULES,
        "",
        OUTPUT_CONTRACT,
    ]
    if extra_instructions:
        parts += ["", "ADDITIONAL INSTRUCTIONS (take priority):", extra_instructions]
    return "\n".join(parts)


def _short(s: Optional[str], n: int) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 3] + "..."


def render_contribution(c: AgentContribution, rationale_chars: int = 400) -> str:
    target = f" -> {c.target_contribution_id}" if c.target_contribution_id else ""
    ev = "; ".join(c.payload.explanation.evidence[:4])
    return (
        f"[id={c.contribution_id}] turn {c.turn_index} | {c.agent_id} | {c.tag.value}{target}\n"
        f"  claim: {_short(c.payload.prediction.claim, 200)} (conf {c.payload.prediction.confidence:.2f})\n"
        f"  why: {_short(c.payload.explanation.rationale, rationale_chars)}"
        + (f"\n  evidence: {_short(ev, 250)}" if ev else "")
    )


def render_board(snapshot: BlackboardSnapshot, max_history: int = 12) -> str:
    history: List[AgentContribution] = snapshot.contributions[-max_history:]
    omitted = len(snapshot.contributions) - len(history)
    lines = [f"TASK ({snapshot.task_id}):", snapshot.task_description.strip()]
    if snapshot.initial_context:
        ctx = json.dumps(snapshot.initial_context, ensure_ascii=False, default=str)
        lines += ["", "CONTEXT:", _short(ctx, 6000)]
    if snapshot.active_claim:
        lines += ["", f"CURRENT LEADING CLAIM: {snapshot.active_claim.claim}"]
    lines += ["", f"BLACKBOARD (status={snapshot.status.value}, {len(snapshot.contributions)} posts):"]
    if omitted:
        lines.append(f"(... {omitted} older posts omitted ...)")
    lines += [render_contribution(c) for c in history] or ["(empty - you are posting the first hypothesis; use REVISE with target null)"]
    return "\n".join(lines)


def build_messages(
    persona: Persona,
    snapshot: BlackboardSnapshot,
    agent_id: str,
    extra_instructions: Optional[str] = None,
    max_history: int = 12,
) -> List[ChatMessage]:
    user = (
        render_board(snapshot, max_history)
        + f"\n\nYou are agent '{agent_id}'. Read the board and post your next contribution as JSON."
    )
    return [
        ChatMessage(role="system", content=system_prompt(persona, extra_instructions)),
        ChatMessage(role="user", content=user),
    ]


def repair_message(bad_output: str, error: str) -> ChatMessage:
    return ChatMessage(
        role="user",
        content=(
            "Your previous output was invalid.\nERROR: "
            + _short(error, 600)
            + "\nReturn ONLY the corrected JSON object matching the required format."
        ),
    )
