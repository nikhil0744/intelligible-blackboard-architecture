"""Deterministic offline backend for tests, CI and S3/S4 development without a GPU."""

from __future__ import annotations

import json
import threading
import time
from typing import Callable, List, Optional, Union

from ..types import LLMError, LLMRequest, LLMResponse

Responder = Callable[[LLMRequest], str]


class MockBackend:
    """Returns scripted outputs.

    - `responses`: list of strings returned in order (cycled when exhausted), or
    - `responder`: function(request) -> str for context-aware fakes.
    Default: a valid opening REVISE decision.
    `fail_first_n` simulates transient failures to exercise broker retries.
    """

    name = "mock"

    def __init__(
        self,
        responses: Optional[List[str]] = None,
        responder: Optional[Responder] = None,
        delay_s: float = 0.0,
        fail_first_n: int = 0,
    ):
        self.responses = responses or []
        self.responder = responder
        self.delay_s = delay_s
        self._fail_left = fail_first_n
        self._i = 0
        self._lock = threading.Lock()
        self.calls: List[LLMRequest] = []
        self.max_in_flight = 0
        self._in_flight = 0

    def complete(self, request: LLMRequest, model: str) -> LLMResponse:
        with self._lock:
            self.calls.append(request)
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
            fail = self._fail_left > 0
            if fail:
                self._fail_left -= 1
        try:
            if self.delay_s:
                time.sleep(self.delay_s)
            if fail:
                raise LLMError("mock transient failure")
            text = self._next(request)
        finally:
            with self._lock:
                self._in_flight -= 1
        prompt_chars = sum(len(m.content) for m in request.messages)
        return LLMResponse(
            text=text,
            model=model,
            backend=self.name,
            prompt_tokens=prompt_chars // 4,
            completion_tokens=len(text) // 4,
            latency_ms=self.delay_s * 1000,
        )

    def _next(self, request: LLMRequest) -> str:
        if self.responder:
            return self.responder(request)
        if self.responses:
            with self._lock:
                out = self.responses[self._i % len(self.responses)]
                self._i += 1
            return out
        return json.dumps(default_decision())


def default_decision(tag: str = "REVISE", claim: str = "mock claim", target: Union[str, None] = None) -> dict:
    return {
        "tag": tag,
        "target_contribution_id": target,
        "prediction": {"claim": claim, "confidence": 0.6, "summary": claim},
        "explanation": {
            "rationale": "Mock rationale.",
            "evidence": ["mock evidence"],
            "assumptions": [],
            "limitations": None,
        },
    }
