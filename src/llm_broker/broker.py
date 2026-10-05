"""Multi-threaded model broker + hardware-resource guard (Student 2).

- A bounded semaphore caps in-flight model calls (a 7B model on a laptop can
  only serve ~1-2 requests at once; raise on a GPU box).
- A thread pool gives non-blocking `submit()` futures and `generate_many()`.
- `agenerate()` lets S1's asyncio scheduler await calls without blocking the loop.
- Retries with exponential backoff; thread-safe token/latency accounting per agent,
  which S4's analytics reads via `usage()` / `usage_by_agent()`.
S3 reuses the same broker instance for counterfactual replays (no second inference path).
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence

from .backends.base import LLMBackend
from .types import LLMError, LLMRequest, LLMResponse, UsageStats


class ModelBroker:
    def __init__(
        self,
        backend: LLMBackend,
        default_model: str,
        max_concurrency: int = 2,
        max_retries: int = 2,
        backoff_s: float = 0.5,
        max_workers: Optional[int] = None,
    ):
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self.backend = backend
        self.default_model = default_model
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self._slots = threading.BoundedSemaphore(max_concurrency)
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers or max(4, max_concurrency * 2), thread_name_prefix="llm"
        )
        self._stats_lock = threading.Lock()
        self._total = UsageStats()
        self._per_agent: Dict[str, UsageStats] = {}

    # -- core ------------------------------------------------------------
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Blocking call; safe to invoke from many threads at once."""
        model = request.model or self.default_model
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 2):
            with self._slots:
                try:
                    resp = self.backend.complete(request, model)
                    resp.attempts = attempt
                    self._record(request.agent_id, resp)
                    return resp
                except LLMError as e:
                    last_err = e
            if attempt <= self.max_retries:
                time.sleep(self.backoff_s * (2 ** (attempt - 1)))
        self._record_failure(request.agent_id)
        raise LLMError(f"{self.backend.name}/{model} failed after {self.max_retries + 1} attempts: {last_err}")

    def submit(self, request: LLMRequest) -> "Future[LLMResponse]":
        return self._pool.submit(self.generate, request)

    def generate_many(self, requests: Sequence[LLMRequest]) -> List[LLMResponse]:
        """Run requests in parallel (bounded by max_concurrency); preserves input order."""
        return [f.result() for f in [self.submit(r) for r in requests]]

    async def agenerate(self, request: LLMRequest) -> LLMResponse:
        return await asyncio.wrap_future(self.submit(request))

    # -- accounting ---------------------------------------------------------
    def _record(self, agent_id: Optional[str], resp: LLMResponse) -> None:
        with self._stats_lock:
            for s in (self._total, self._per_agent.setdefault(agent_id or "_anon", UsageStats())):
                s.calls += 1
                s.prompt_tokens += resp.prompt_tokens
                s.completion_tokens += resp.completion_tokens
                s.total_latency_ms += resp.latency_ms

    def _record_failure(self, agent_id: Optional[str]) -> None:
        with self._stats_lock:
            self._total.failures += 1
            self._per_agent.setdefault(agent_id or "_anon", UsageStats()).failures += 1

    def usage(self) -> UsageStats:
        with self._stats_lock:
            return self._total.model_copy()

    def usage_by_agent(self) -> Dict[str, UsageStats]:
        with self._stats_lock:
            return {k: v.model_copy() for k, v in self._per_agent.items()}

    def reset_usage(self) -> None:
        with self._stats_lock:
            self._total = UsageStats()
            self._per_agent.clear()

    # -- lifecycle ------------------------------------------------------------
    def close(self) -> None:
        self._pool.shutdown(wait=True)

    def __enter__(self) -> "ModelBroker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
