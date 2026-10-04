"""Prompt matrix (Student 2): personas, PEX output schema, templates, parser."""

from .budget import PromptBudgetReport, build_prompt, estimate_tokens
from .parser import DecisionParseError, extract_json, parse_decision
from .personas import Persona, domains, get_persona, list_personas
from .schema import AgentDecision, decision_json_schema
from .templates import build_messages, render_board, repair_message, system_prompt
from .validation import DecisionSemanticError, normalize_claim, validate_decision

__all__ = [
    "AgentDecision",
    "DecisionParseError",
    "DecisionSemanticError",
    "Persona",
    "PromptBudgetReport",
    "build_messages",
    "build_prompt",
    "decision_json_schema",
    "domains",
    "estimate_tokens",
    "extract_json",
    "get_persona",
    "list_personas",
    "normalize_claim",
    "parse_decision",
    "render_board",
    "repair_message",
    "system_prompt",
    "validate_decision",
]
