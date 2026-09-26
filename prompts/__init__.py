"""Prompt matrix (Student 2): personas, PEX output schema, templates, parser."""

from .parser import DecisionParseError, extract_json, parse_decision
from .personas import Persona, domains, get_persona, list_personas
from .schema import AgentDecision, decision_json_schema
from .templates import build_messages, render_board, repair_message, system_prompt

__all__ = [
    "AgentDecision",
    "DecisionParseError",
    "Persona",
    "build_messages",
    "decision_json_schema",
    "domains",
    "extract_json",
    "get_persona",
    "list_personas",
    "parse_decision",
    "render_board",
    "repair_message",
    "system_prompt",
]
