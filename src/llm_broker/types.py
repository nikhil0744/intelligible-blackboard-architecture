"""Shared request/response types for the local LLM broker (Student 2)."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMRequest(BaseModel):
    """One model call. `json_schema` asks the backend for schema-constrained JSON output."""

    messages: List[ChatMessage]
    model: Optional[str] = Field(default=None, description="Overrides the broker default model.")
    temperature: float = 0.3
    max_tokens: int = 1024
    seed: Optional[int] = None
    json_schema: Optional[Dict[str, Any]] = None
    agent_id: Optional[str] = Field(default=None, description="For per-agent usage accounting.")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    text: str
    model: str
    backend: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    attempts: int = 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMError(RuntimeError):
    """Raised when a backend call fails after all retries."""


class UsageStats(BaseModel):
    calls: int = 0
    failures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_latency_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens
