"""Optional per-trial inference limits, propagated with the broker's context."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import threading
import time

from .types import LLMError


class BudgetExceeded(LLMError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


@dataclass
class CallBudget:
    max_attempts: int = 100
    timeout_s: float = 1200
    attempts: int = 0
    started: float = field(default_factory=time.monotonic)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def remaining(self) -> float:
        return max(0.0, self.timeout_s - (time.monotonic() - self.started))

    def check(self) -> None:
        if self.remaining() <= 0:
            raise BudgetExceeded("time_limit")
        if self.attempts >= self.max_attempts:
            raise BudgetExceeded("call_limit")

    def take(self) -> float:
        with self._lock:
            self.check()
            self.attempts += 1
            return self.remaining()


_budget: ContextVar[CallBudget | None] = ContextVar("call_budget", default=None)


def current_budget() -> CallBudget | None:
    return _budget.get()


@contextmanager
def budget_scope(budget: CallBudget):
    token = _budget.set(budget)
    try:
        yield budget
    finally:
        _budget.reset(token)
