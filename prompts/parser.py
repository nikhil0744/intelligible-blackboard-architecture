"""Robust extraction of the agent's JSON from raw 7B-model text."""

from __future__ import annotations

import json
import re
from typing import Any, Dict

from pydantic import ValidationError

from .schema import AgentDecision

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class DecisionParseError(ValueError):
    pass


def extract_json(text: str) -> Dict[str, Any]:
    """Return the first JSON object in `text` (handles fences, prose, trailing commas)."""
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for cand in candidates:
        start = cand.find("{")
        while start != -1:
            depth, in_str, esc = 0, False, False
            for i in range(start, len(cand)):
                ch = cand[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                elif ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        blob = cand[start : i + 1]
                        for attempt in (blob, re.sub(r",\s*([}\]])", r"\1", blob)):
                            try:
                                obj = json.loads(attempt)
                                if isinstance(obj, dict):
                                    return obj
                            except json.JSONDecodeError:
                                pass
                        break
            start = cand.find("{", start + 1)
    raise DecisionParseError("no JSON object found in model output")


def parse_decision(text: str) -> AgentDecision:
    obj = extract_json(text)
    # tolerate common small-model slips
    if "prediction" not in obj and "claim" in obj:
        obj["prediction"] = {"claim": obj.pop("claim"), "confidence": obj.pop("confidence", 0.5)}
    if isinstance(obj.get("explanation"), str):
        obj["explanation"] = {"rationale": obj["explanation"], "evidence": []}
    exp = obj.get("explanation")
    if isinstance(exp, dict) and isinstance(exp.get("evidence"), str):
        exp["evidence"] = [exp["evidence"]]
    pred = obj.get("prediction")
    if isinstance(pred, dict) and isinstance(pred.get("confidence"), (int, float)) and pred["confidence"] > 1:
        pred["confidence"] = min(pred["confidence"] / 100.0, 1.0)  # "85" -> 0.85
    try:
        return AgentDecision.model_validate(obj)
    except ValidationError as e:
        raise DecisionParseError(str(e)) from e
