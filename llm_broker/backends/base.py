from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import LLMRequest, LLMResponse


@runtime_checkable
class LLMBackend(Protocol):
    """A synchronous, thread-safe model backend. The broker adds concurrency and retries."""

    name: str

    def complete(self, request: LLMRequest, model: str) -> LLMResponse:
        ...
