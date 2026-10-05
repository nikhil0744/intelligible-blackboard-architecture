"""The JSON an agent must emit each turn: PXP tag + PEX (prediction + explanation).

`AgentDecision` is deliberately flatter than S1's `AgentContribution` (the model
never invents ids, timestamps, or turn indices); `agents.base` maps it onto the contract.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from contracts.schemas import Explanation, PEXPayload, Prediction, PXPTag


class DecisionPrediction(BaseModel):
    model_config = ConfigDict(extra="ignore")
    claim: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    summary: Optional[str] = None


class DecisionExplanation(BaseModel):
    model_config = ConfigDict(extra="ignore")
    rationale: str = Field(..., min_length=1)
    evidence: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    limitations: Optional[str] = None


class AgentDecision(BaseModel):
    model_config = ConfigDict(extra="ignore")

    tag: PXPTag
    target_contribution_id: Optional[str] = None
    prediction: DecisionPrediction
    explanation: DecisionExplanation

    @field_validator("tag", mode="before")
    @classmethod
    def _upper(cls, v: Any) -> Any:
        return v.strip().upper() if isinstance(v, str) else v

    @field_validator("target_contribution_id", mode="before")
    @classmethod
    def _blank_to_none(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip().lower() in {"", "none", "null", "n/a"}:
            return None
        return v

    def to_pex(self) -> PEXPayload:
        p, e = self.prediction, self.explanation
        return PEXPayload(
            prediction=Prediction(claim=p.claim, confidence=p.confidence, summary=p.summary),
            explanation=Explanation(
                rationale=e.rationale,
                evidence=e.evidence,
                assumptions=e.assumptions,
                limitations=e.limitations,
            ),
        )


def decision_json_schema(reasoning_first: bool = False) -> Dict[str, Any]:
    """Compact, self-contained JSON schema for constrained decoding (Ollama `format`)."""
    schema = {
        "type": "object",
        "properties": {
            "tag": {"type": "string", "enum": [t.value for t in PXPTag]},
            "target_contribution_id": {"type": ["string", "null"]},
            "prediction": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "summary": {"type": ["string", "null"]},
                },
                "required": ["claim", "confidence"],
            },
            "explanation": {
                "type": "object",
                "properties": {
                    "rationale": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                    "assumptions": {"type": "array", "items": {"type": "string"}},
                    "limitations": {"type": ["string", "null"]},
                },
                "required": ["rationale", "evidence"],
            },
        },
        "required": ["tag", "target_contribution_id", "prediction", "explanation"],
    }
    if reasoning_first:
        # Ordering is a decoding/prompt hint, not a semantic guarantee. Keep the
        # same fields and validation contract while placing reasoning first.
        order = ("explanation", "prediction", "tag", "target_contribution_id")
        schema["properties"] = {key: schema["properties"][key] for key in order}
        schema["required"] = list(order)
    return schema
