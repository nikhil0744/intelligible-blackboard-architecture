"""Loads the persona matrix (prompts/personas.json)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel

from contracts.schemas import AgentRole

PERSONA_FILE = Path(__file__).with_name("personas.json")


class Persona(BaseModel):
    id: str
    domain: str
    role: AgentRole
    title: str
    expertise: str
    stance: str


@lru_cache(maxsize=1)
def _load(path: str = str(PERSONA_FILE)) -> Dict[str, Persona]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out: Dict[str, Persona] = {}
    for row in raw["personas"]:
        p = Persona(**row)
        if p.id in out:
            raise ValueError(f"duplicate persona id {p.id}")
        out[p.id] = p
    return out


def get_persona(persona_id: str) -> Persona:
    try:
        return _load()[persona_id]
    except KeyError:
        raise KeyError(f"unknown persona {persona_id!r}; known: {sorted(_load())}") from None


def list_personas(domain: Optional[str] = None) -> List[Persona]:
    return [p for p in _load().values() if domain is None or p.domain == domain]


def domains() -> List[str]:
    return sorted({p.domain for p in _load().values()})
