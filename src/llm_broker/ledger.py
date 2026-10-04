"""Per-call usage ledger (task S2.5).

Every model call made through the broker becomes one `CallRecord`: who called, in
which trial and phase, which model, how many tokens, how long, how many backend
attempts, and whether it failed. Per-trial cost is computed from that trial's own
records, never from the broker's running total.

Tagging a call:
- explicitly, via `LLMRequest.metadata` keys `trial_id`, `session_id`, `phase`, `is_repair`;
- or implicitly, for every call made inside `with usage_scope(trial_id=..., phase=...)`.

Phases: "deliberation" (ordinary turns, the default), "baseline_replay",
"candidate_replay", "tool", "evaluator". Repairs keep the phase of the call they
repair and set `is_repair`.
"""

from __future__ import annotations

import contextvars
import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from .types import CallRecord, UsageSummary

PHASES = ("deliberation", "baseline_replay", "candidate_replay", "tool", "evaluator")
DEFAULT_PHASE = "deliberation"

_scope: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar("llm_usage_scope", default={})


@contextmanager
def usage_scope(
    trial_id: Optional[str] = None,
    phase: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Iterator[None]:
    """Tag every broker call made in this block. Nested scopes override only what they set.

    The scope follows `ModelBroker.submit()` and `agents.act_parallel()` into their worker
    threads. A thread you start yourself needs `contextvars.copy_context().run(...)`.
    """
    if phase is not None and phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}; expected one of {PHASES}")
    new = dict(_scope.get())
    for k, v in (("trial_id", trial_id), ("phase", phase), ("session_id", session_id)):
        if v is not None:
            new[k] = v
    token = _scope.set(new)
    try:
        yield
    finally:
        _scope.reset(token)


def current_scope() -> Dict[str, Any]:
    return dict(_scope.get())


class UsageLedger:
    """Thread-safe, append-only list of CallRecords, optionally mirrored to a JSONL file."""

    def __init__(self, path: Optional[Union[str, Path]] = None):
        self._lock = threading.Lock()
        self._records: List[CallRecord] = []
        self._path: Optional[Path] = None
        if path is not None:
            self.attach_file(path)

    def attach_file(self, path: Union[str, Path]) -> None:
        """Append every later record to `path` as it happens, so an interrupted run keeps its ledger."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._path = p

    def append(self, record: CallRecord) -> None:
        with self._lock:
            self._records.append(record)
            if self._path is not None:
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(record.model_dump_json() + "\n")

    def records(
        self,
        trial_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        phase: Optional[str] = None,
    ) -> List[CallRecord]:
        with self._lock:
            out = list(self._records)
        if trial_id is not None:
            out = [r for r in out if r.trial_id == trial_id]
        if agent_id is not None:
            out = [r for r in out if r.agent_id == agent_id]
        if phase is not None:
            out = [r for r in out if r.phase == phase]
        return out

    def summary(self, **filters: Any) -> UsageSummary:
        return UsageSummary.from_records(self.records(**filters))

    def by_phase(self, trial_id: Optional[str] = None) -> Dict[str, UsageSummary]:
        recs = self.records(trial_id=trial_id)
        return {p: UsageSummary.from_records([r for r in recs if r.phase == p]) for p in sorted({r.phase for r in recs})}

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    def write_jsonl(self, path: Union[str, Path], trial_id: Optional[str] = None) -> int:
        recs = self.records(trial_id=trial_id)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r.model_dump(mode="json"), ensure_ascii=False) + "\n")
        return len(recs)

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)
