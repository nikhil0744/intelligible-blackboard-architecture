"""Single-writer records with verified per-attempt commits; no database on Drive."""
from __future__ import annotations

import csv
import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from llm_broker.types import CallRecord, UsageSummary

from .preparation import digest, fingerprint


def now():
    return datetime.now(timezone.utc).isoformat()


class PersistenceError(RuntimeError):
    pass


def write_json(path: Path, value):
    """Replace atomically, flush, and read back; a failure always stops the batch."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if json.loads(path.read_text()) != value:
            raise OSError("record read-back differs from write")
    except OSError as exc:
        raise PersistenceError(f"Cannot safely save {path}: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)


class Attempt:
    def __init__(self, path: Path, identity: dict):
        self.path, self.identity = path, identity
        path.mkdir()
        write_json(path / "attempt.json", identity)
        for name in ("events.jsonl", "calls.jsonl"):
            (path / name).touch(exist_ok=False)

    def event(self, kind, **payload):
        record = {"time": now(), "kind": kind, **payload}
        try:
            with (self.path / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            raise PersistenceError(f"Cannot save event: {exc}") from exc

    def freeze(self, output):
        path = self.path / "final_output.json"
        if path.exists():
            raise PersistenceError("final output was already frozen")
        write_json(path, {"output": output, "frozen_at": now()})
        self.event("output_frozen", sha256=digest(path))
        return json.loads(path.read_text())["output"]

    def finish(self, result):
        if (self.path / "complete.json").exists():
            raise PersistenceError("attempt is already committed")
        result = {**self.identity, **result, "finished_at": now()}
        write_json(self.path / "result.json", result)
        checksums = {}
        for name in ("attempt.json", "events.jsonl", "calls.jsonl", "final_output.json", "result.json"):
            path = self.path / name
            if path.exists():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
                checksums[name] = digest(path)
        write_json(self.path / "complete.json", {"sha256": checksums})
        return read_result(self.path)


def read_result(path: Path):
    try:
        marker = json.loads((path / "complete.json").read_text())
        required = {"attempt.json", "result.json", "events.jsonl", "calls.jsonl"}
        if not required <= set(marker["sha256"]):
            raise ValueError("missing committed records")
        for name, expected in marker["sha256"].items():
            if Path(name).name != name or digest(path / name) != expected:
                raise ValueError(f"checksum mismatch: {name}")
        return json.loads((path / "result.json").read_text())
    except (OSError, ValueError, KeyError) as exc:
        raise PersistenceError(f"Invalid completed attempt {path}: {exc}") from exc


class RunStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.selection = json.loads((self.root / "selection.json").read_text())
        if fingerprint(self.manifest["configuration"]) != self.manifest["configuration_hash"]:
            raise PersistenceError("run manifest fingerprint is invalid")
        if fingerprint(self.selection) != self.manifest["selection_hash"]:
            raise PersistenceError("saved selection was changed")

    @classmethod
    def create(cls, root: Path, configuration: dict, selection: list[dict], observations: dict):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=False)
        (root / "attempts").mkdir()
        write_json(root / "selection.json", selection)
        write_json(root / "manifest.json", {"schema_version": 1, "run_id": root.name,
                   "created_at": now(), "configuration": configuration,
                   "configuration_hash": fingerprint(configuration), "selection_hash": fingerprint(selection),
                   "observations": observations})
        return cls(root)

    @contextmanager
    def writer(self):
        # Local advisory lock: one notebook/runtime may write a Drive run at a time.
        # This is deliberately not a distributed lock between two Colab runtimes.
        locks = Path(tempfile.gettempdir()) / "blackboard-benchmark-locks"
        locks.mkdir(exist_ok=True)
        with (locks / fingerprint(str(self.root.resolve()))).open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise PersistenceError("Another process in this runtime is writing this run") from exc
            try:
                yield self
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def validate(self, configuration):
        if fingerprint(configuration) != self.manifest["configuration_hash"]:
            raise ValueError("Cannot resume: code, data, model digest, settings or budgets changed. Use the pinned code/model and original settings, or create a new run.")

    def start(self, task_id):
        identity = {"run_id": self.root.name, "attempt_id": uuid4().hex, "session_id": uuid4().hex,
                    "task_id": task_id, "started_at": now()}
        return Attempt(self.root / "attempts" / identity["attempt_id"], identity)

    def attempts(self):
        records = []
        for path in sorted((self.root / "attempts").iterdir()):
            if not path.is_dir():
                continue
            if (path / "complete.json").exists():
                records.append(read_result(path))
            elif (path / "attempt.json").exists():
                records.append({**json.loads((path / "attempt.json").read_text()),
                                "status": "interrupted", "committed": False})
        return sorted(records, key=lambda r: (r["started_at"], r["attempt_id"]))

    def summarize(self):
        records = self.attempts()
        latest = {r["task_id"]: r for r in records}
        selected = [task["task_id"] for task in self.selection]
        outcomes = [latest.get(tid, {"task_id": tid, "status": "pending"}) for tid in selected]
        scored = [r for r in outcomes if r.get("grade", {}).get("status") == "scored"]
        correct = sum(r.get("grade", {}).get("correct") is True for r in scored)
        completed = sum(r["status"] in {"consensus", "deadlock", "turn_limit", "tool_limit", "call_limit", "time_limit"} for r in outcomes)
        call_records = []
        partial_calls = 0
        for attempt in records:
            path = self.root / "attempts" / attempt["attempt_id"] / "calls.jsonl"
            if not path.exists() and attempt.get("committed") is False:
                # The runtime may die between writing attempt.json and creating its
                # ledgers. Keep that interrupted attempt without blocking a restart.
                partial_calls += 1
                continue
            for line in path.read_text().splitlines():
                try:
                    call_records.append(CallRecord.model_validate_json(line))
                except ValueError:
                    if attempt.get("committed", True):
                        raise PersistenceError(f"Invalid call ledger {path}")
                    partial_calls += 1
        usage = UsageSummary.from_records(call_records)
        interrupted_attempts = sum(r["status"] == "interrupted" for r in records)
        summary = {"run_id": self.root.name, "selected": len(selected), "started": len(latest),
                   "completed": completed, "scored": len(scored), "correct": correct,
                   "incorrect": len(scored) - correct, "ungraded": len(outcomes) - len(scored),
                   "timed_out": sum(r["status"] == "time_limit" for r in outcomes),
                   "errored": sum(r["status"] in {"error", "invalid_output", "model_failure", "evaluation_error"} for r in outcomes),
                   "interrupted": sum(r["status"] == "interrupted" for r in outcomes),
                   "pending": sum(r["status"] == "pending" for r in outcomes),
                   "skipped": sum(r["status"] == "skipped" for r in outcomes),
                   "attempts": len(records), "partial_call_records": partial_calls,
                   "consensus": sum(bool(r.get("consensus")) for r in outcomes),
                   "success_rate_all_selected": correct / len(selected) if selected else None,
                   "success_rate_scored": correct / len(scored) if scored else None,
                   "aggregate_policy": "latest explicit attempt per task; all attempts retained and included in usage",
                   "official_aggregate": False,
                   "usage_all_attempts": {**usage.model_dump(),
                       "interrupted_attempts": interrupted_attempts,
                       "total_tokens": None if interrupted_attempts or partial_calls else usage.total_tokens},
                   "outcomes": outcomes}
        write_json(self.root / "summary.json", summary)
        # CSV is derived, never the authoritative record.
        temporary = self.root / ".results.csv.tmp"
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["task_id", "attempt_id", "status", "consensus", "score", "correct", "elapsed_s", "total_tokens"])
            writer.writeheader()
            for r in outcomes:
                writer.writerow({"task_id": r["task_id"], "attempt_id": r.get("attempt_id"), "status": r["status"],
                                 "consensus": r.get("consensus"), "score": r.get("grade", {}).get("score"),
                                 "correct": r.get("grade", {}).get("correct"), "elapsed_s": r.get("elapsed_s"),
                                 "total_tokens": r.get("usage", {}).get("total_tokens")})
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.root / "results.csv")
        return summary
