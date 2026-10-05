"""Shared request/response types for the local LLM broker (Student 2)."""

from __future__ import annotations

from datetime import datetime
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
    timeout_s: Optional[float] = Field(default=None, gt=0, description="Optional request deadline override.")
    json_schema: Optional[Dict[str, Any]] = None
    agent_id: Optional[str] = Field(default=None, description="For per-agent usage accounting.")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Ledger tags: trial_id, session_id, phase, is_repair, estimated_prompt_tokens.",
    )


class LLMResponse(BaseModel):
    text: str
    model: str
    backend: str
    # None means the backend did not report the count. Unknown usage is never treated as zero.
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    latency_ms: float = 0.0
    attempts: int = 1

    @property
    def usage_known(self) -> bool:
        return self.prompt_tokens is not None and self.completion_tokens is not None

    @property
    def total_tokens(self) -> Optional[int]:
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None
        return self.prompt_tokens + self.completion_tokens


class LLMError(RuntimeError):
    """Raised when a backend call fails after all retries."""


class UsageStats(BaseModel):
    """Running totals since the broker was created or reset. Not a per-trial figure:
    use `ModelBroker.trial_usage(trial_id)` for that. Token sums cover known usage only."""

    calls: int = 0
    failures: int = 0
    unknown_usage_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_latency_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class CallRecord(BaseModel):
    """One broker call (including its backend retries) in the usage ledger."""

    call_id: str
    started_at: datetime
    trial_id: Optional[str] = None
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    phase: str = "deliberation"
    is_repair: bool = False
    model: str
    backend: str
    ok: bool = True
    error: Optional[str] = None
    attempts: int = 1  # backend attempts, including the successful one
    prompt_tokens: Optional[int] = None  # None = the backend did not report it
    completion_tokens: Optional[int] = None
    estimated_prompt_tokens: Optional[int] = None  # the prompt builder's estimate, for auditing the budget
    latency_ms: float = 0.0
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    seed: Optional[int] = None

    @property
    def usage_known(self) -> bool:
        return self.prompt_tokens is not None and self.completion_tokens is not None


class UsageSummary(BaseModel):
    """Totals over a set of CallRecords. Token sums cover only calls whose usage is known."""

    calls: int = 0
    failures: int = 0
    repairs: int = 0
    backend_attempts: int = 0
    unknown_usage_calls: int = 0
    unreported_retry_attempts: int = 0
    known_prompt_tokens: int = 0
    known_completion_tokens: int = 0
    total_latency_ms: float = 0.0

    @property
    def known_total_tokens(self) -> int:
        return self.known_prompt_tokens + self.known_completion_tokens

    @property
    def total_tokens(self) -> Optional[int]:
        """Exact total, or None when any call's usage is unknown (unknown is not zero)."""
        return None if self.unknown_usage_calls or self.unreported_retry_attempts else self.known_total_tokens

    @classmethod
    def from_records(cls, records: List["CallRecord"]) -> "UsageSummary":
        s = cls()
        for r in records:
            s.calls += 1
            s.failures += 0 if r.ok else 1
            s.repairs += 1 if r.is_repair else 0
            s.backend_attempts += r.attempts
            # A successful retry reports only its own tokens. Earlier failed attempts
            # may have consumed tokens on the server but returned no usage.
            if r.ok:
                s.unreported_retry_attempts += max(0, r.attempts - 1)
            s.total_latency_ms += r.latency_ms
            if r.usage_known:
                s.known_prompt_tokens += r.prompt_tokens or 0
                s.known_completion_tokens += r.completion_tokens or 0
            else:
                s.unknown_usage_calls += 1
        return s
