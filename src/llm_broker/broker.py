"""Multi-threaded model broker + hardware-resource guard (Student 2).

- A bounded semaphore caps in-flight model calls (a 7B model on a laptop can
  only serve ~1-2 requests at once; raise on a GPU box).
- A thread pool gives non-blocking `submit()` futures and `generate_many()`.
- `agenerate()` lets S1's asyncio scheduler await calls without blocking the loop.
- Retries with exponential backoff.
- Every call is written to `self.ledger` (see `ledger.py`): per-trial cost comes from
  `trial_usage(trial_id)`, never from the running totals in `usage()`.
- One broker serves a whole batch: `close()` only releases the worker threads, and the
  broker stays usable afterwards (the pool is recreated on demand).
S3 reuses the same broker instance for counterfactual replays (no second inference path).
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
import time
import uuid
from datetime import datetime, timezone
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Sequence

from .backends.base import LLMBackend
from .ledger import DEFAULT_PHASE, UsageLedger, current_scope
from .types import CallRecord, LLMError, LLMRequest, LLMResponse, UsageStats, UsageSummary
from .budget import BudgetExceeded, current_budget


class ModelBroker:
    def __init__(
        self,
        backend: LLMBackend,
        default_model: str,
        max_concurrency: int = 2,
        max_retries: int = 2,
        backoff_s: float = 0.5,
        max_workers: Optional[int] = None,
        ledger: Optional[UsageLedger] = None,
        response_hook: Optional[Callable[[LLMRequest, LLMResponse], None]] = None,
    ):
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self.backend = backend
        self.default_model = default_model
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self._slots = threading.BoundedSemaphore(max_concurrency)
        self._max_workers = max_workers or max(4, max_concurrency * 2)
        self._pool: Optional[ThreadPoolExecutor] = None
        self._pool_lock = threading.Lock()
        self.ledger = ledger if ledger is not None else UsageLedger()
        self.response_hook = response_hook
        self.settings = None  # InferenceSettings when built by llm_broker.config
        self._stats_lock = threading.Lock()
        self._total = UsageStats()
        self._per_agent: Dict[str, UsageStats] = {}

    # -- core ------------------------------------------------------------
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Blocking call; safe to invoke from many threads at once."""
        model = request.model or self.default_model
        started_at = datetime.now(timezone.utc)
        t0 = time.perf_counter()
        last_err: Optional[Exception] = None
        attempts = 0
        logged = False
        budget = current_budget()
        try:
            for attempt in range(1, self.max_retries + 2):
                with self._slots:
                    effective_request = request
                    if budget is not None:
                        remaining = budget.take()
                        effective_request = request.model_copy(update={
                            "timeout_s": min(request.timeout_s or remaining, remaining),
                        })
                    attempts = attempt
                    try:
                        resp = self.backend.complete(effective_request, model)
                    except BudgetExceeded:
                        raise
                    except LLMError as e:
                        last_err = e
                    else:
                        resp.attempts = attempt
                        self._record(request.agent_id, resp)
                        self._log_call(request, model, started_at, attempt, (time.perf_counter() - t0) * 1000, resp=resp)
                        logged = True
                        if self.response_hook is not None:
                            self.response_hook(request, resp)
                        return resp
                if attempt <= self.max_retries:
                    delay = self.backoff_s * (2 ** (attempt - 1))
                    if budget is not None:
                        budget.check()
                        if delay >= budget.remaining():
                            raise BudgetExceeded("time_limit")
                    time.sleep(delay)
        except BaseException as e:
            self._record_failure(request.agent_id)
            # Interrupted calls have unknown usage. Denied calls consume zero backend attempts.
            if not logged:
                self._log_call(request, model, started_at, attempts, (time.perf_counter() - t0) * 1000, error=str(e))
            raise
        self._record_failure(request.agent_id)
        self._log_call(request, model, started_at, attempts, (time.perf_counter() - t0) * 1000, error=str(last_err))
        raise LLMError(f"{self.backend.name}/{model} failed after {attempts} attempts: {last_err}")

    def submit(self, request: LLMRequest) -> "Future[LLMResponse]":
        # carry the caller's usage_scope (trial / phase) into the worker thread
        ctx = contextvars.copy_context()
        return self._get_pool().submit(ctx.run, self.generate, request)

    def _get_pool(self) -> ThreadPoolExecutor:
        with self._pool_lock:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(max_workers=self._max_workers, thread_name_prefix="llm")
            return self._pool

    def generate_many(self, requests: Sequence[LLMRequest]) -> List[LLMResponse]:
        """Run requests in parallel (bounded by max_concurrency); preserves input order."""
        return [f.result() for f in [self.submit(r) for r in requests]]

    async def agenerate(self, request: LLMRequest) -> LLMResponse:
        return await asyncio.wrap_future(self.submit(request))

    # -- accounting ---------------------------------------------------------
    def _log_call(
        self,
        request: LLMRequest,
        model: str,
        started_at: datetime,
        attempts: int,
        wall_ms: float,
        resp: Optional[LLMResponse] = None,
        error: Optional[str] = None,
    ) -> None:
        tags = {**current_scope(), **request.metadata}  # explicit request tags win over the scope
        self.ledger.append(
            CallRecord(
                call_id=uuid.uuid4().hex,
                started_at=started_at,
                trial_id=tags.get("trial_id"),
                session_id=tags.get("session_id"),
                agent_id=request.agent_id,
                phase=tags.get("phase") or DEFAULT_PHASE,
                is_repair=bool(tags.get("is_repair", False)),
                model=resp.model if resp else model,
                backend=self.backend.name,
                ok=resp is not None,
                error=error,
                attempts=attempts,
                prompt_tokens=resp.prompt_tokens if resp else None,
                completion_tokens=resp.completion_tokens if resp else None,
                estimated_prompt_tokens=tags.get("estimated_prompt_tokens"),
                latency_ms=wall_ms,  # wall time including failed attempts and backoff
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                seed=request.seed,
            )
        )

    def _record(self, agent_id: Optional[str], resp: LLMResponse) -> None:
        with self._stats_lock:
            for s in (self._total, self._per_agent.setdefault(agent_id or "_anon", UsageStats())):
                s.calls += 1
                if resp.usage_known:
                    s.prompt_tokens += resp.prompt_tokens or 0
                    s.completion_tokens += resp.completion_tokens or 0
                else:
                    s.unknown_usage_calls += 1
                s.total_latency_ms += resp.latency_ms

    def _record_failure(self, agent_id: Optional[str]) -> None:
        with self._stats_lock:
            self._total.failures += 1
            self._per_agent.setdefault(agent_id or "_anon", UsageStats()).failures += 1

    def trial_usage(self, trial_id: str) -> UsageSummary:
        """Cost of one trial, from that trial's own ledger records only."""
        return self.ledger.summary(trial_id=trial_id)

    def usage(self) -> UsageStats:
        """Running totals for the whole broker lifetime. Do NOT report this as a trial's cost."""
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
        """Release worker threads. The broker remains usable; the ledger is kept."""
        with self._pool_lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True)

    def __enter__(self) -> "ModelBroker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
