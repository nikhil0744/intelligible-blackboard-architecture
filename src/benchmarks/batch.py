"""Deterministic selection, readiness, persistent batches and task-level resume."""
from __future__ import annotations

import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, replace
from importlib import metadata
from pathlib import Path
from uuid import uuid4

from llm_broker.preflight import run_preflight

from .persistence import PersistenceError, RunStore, now
from .preparation import PreparedBenchmark, digest, fingerprint
from .runtime import Limits, ROSTER, SessionRunner


def source_identity():
    source = Path(__file__).resolve().parents[1]
    hashes = {str(path.relative_to(source)): digest(path)
              for path in sorted(source.rglob("*")) if path.is_file()
              and path.suffix in {".py", ".json"} and "__pycache__" not in path.parts}
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source, capture_output=True,
                                  text=True, check=True, timeout=5).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=source,
                                    capture_output=True, text=True, check=True, timeout=5).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        revision, dirty = None, None
    return {"revision": revision, "dirty": dirty, "source_sha256": fingerprint(hashes)}


def package_identity():
    return {"python": platform.python_version(), **{name: metadata.version(name)
            for name in ("pandas", "numpy", "pydantic")}}


def check_readiness(broker, settings, cpu_smoke=False):
    if settings.backend != "ollama" or broker.backend.name != "ollama":
        raise ValueError("The real benchmark command requires local Ollama; no mock or paid-provider fallback is enabled")
    if settings.max_concurrency != 1:
        raise ValueError("The Colab profile requires one model call at a time")
    report = run_preflight(broker, settings, probe=True)
    if not report["ok"]:
        raise ValueError(f"Inference is not ready: {report['checks']}")
    if not report.get("model_digest"):
        raise ValueError("Ollama did not report a model digest; this run cannot be reproduced")
    from .preparation import resources
    preset = resources()["runtime"]
    if settings.model == preset["model"] and report["model_digest"].removeprefix("sha256:") != preset["model_digest"]:
        raise ValueError("The Qwen preset digest differs from the pinned model. Restore the pinned model before this benchmark run.")
    if not cpu_smoke:
        if not report["hardware"]["gpus"]:
            raise ValueError("No GPU assigned. In Colab choose Runtime > Change runtime type > GPU, then rerun setup. CPU smoke requires explicit opt-in.")
        loaded = report.get("loaded_model", {})
        if not loaded.get("size_vram_bytes"):
            raise ValueError("The requested model is not using GPU memory. Check Ollama's GPU setup or explicitly select CPU smoke mode.")
        if not next((c["ok"] for c in report["checks"] if c["name"] == "single_loaded_model"), False):
            raise ValueError("Other models are loaded in Ollama; unload them before starting this single-model profile")
    return report


def configuration(prepared, settings, limits, seed, report, cpu_smoke=False):
    return {"benchmark": "kramabench", "resources": prepared.spec,
            "model_digest": report["model_digest"], "backend_version": report.get("backend_version"),
            "inference": settings.to_dict(), "limits": asdict(limits), "seed": seed,
            "roster": [dict(agent_id=a, persona=p, counterfactual_capable=False) for a, p in ROSTER],
            "recovery_mode": "disabled", "cpu_smoke": cpu_smoke,
            "code_sha256": source_identity()["source_sha256"], "packages": package_identity(),
            "grading": "pinned Success metric; selected numeric_exact tasks; no official aggregate"}


def select_tasks(prepared, limit=3, task_ids=None):
    tasks = {task.task_id: task for task in prepared.tasks()}
    selected = list(tasks) if task_ids is None else list(task_ids)
    if len(set(selected)) != len(selected) or not selected or any(tid not in tasks for tid in selected):
        raise ValueError(f"Choose unique eligible task IDs from {list(tasks)}")
    if task_ids is None:
        if not 1 <= limit <= len(selected):
            raise ValueError(f"This profile contains {len(selected)} tasks; limit must be 1–{len(selected)}")
        selected = selected[:limit]
    return [tasks[tid] for tid in selected]


def validate_destination(results_dir, temporary=False):
    path = Path(results_dir).expanduser().resolve()
    # A Drive path left behind after disconnect is not proof of a mounted Drive.
    in_colab = Path("/content").exists() and bool("google.colab" in sys.modules or
                    os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_BACKEND_VERSION"))
    if in_colab and not temporary:
        drive = Path("/content/drive")
        if not os.path.ismount(drive) or not path.is_relative_to(drive / "MyDrive"):
            raise PersistenceError("Mount Google Drive and choose a folder under /content/drive/MyDrive, or explicitly select temporary smoke mode")
    path.mkdir(parents=True, exist_ok=True)
    probe = path / f".write-probe-{uuid4().hex}"
    try:
        with probe.open("x") as stream:
            stream.write("blackboard")
            stream.flush()
            os.fsync(stream.fileno())
        if probe.read_text() != "blackboard":
            raise PersistenceError("save location failed read-back")
    finally:
        probe.unlink(missing_ok=True)
    return path


def run_batch(prepared: PreparedBenchmark, broker, settings, results_dir, *, limits=None,
              seed=0, limit=3, task_ids=None, cpu_smoke=False, temporary=False,
              observer=print, readiness=None):
    limits = limits or Limits()
    destination = validate_destination(results_dir, temporary)
    selected = select_tasks(prepared, limit, task_ids)
    report = readiness if readiness is not None else check_readiness(broker, settings, cpu_smoke)
    config = configuration(prepared, settings, limits, seed, report, cpu_smoke)
    store = RunStore.create(destination / (time.strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:8]),
                            config, [t.model_dump(mode="json") for t in selected],
                            {"readiness": report, "code": source_identity(), "coverage": prepared.coverage(),
                             "storage": "temporary" if temporary else "persistent", "execution_mode": broker.backend.name})
    observer({"kind": "run_created", "run_dir": str(store.root), "tasks": len(selected)})
    return execute_pending(store, prepared, broker, settings, limits, seed, observer)


def resume_batch(run_dir, prepared, broker, settings, *, cpu_smoke=False,
                 retry_failures=False, observer=print, readiness=None):
    store = RunStore(run_dir)
    saved = store.manifest["configuration"]
    limits = Limits(**saved["limits"])
    report = readiness if readiness is not None else check_readiness(broker, settings, cpu_smoke)
    store.validate(configuration(prepared, settings, limits, saved["seed"], report, cpu_smoke))
    validate_destination(store.root.parent, temporary=store.manifest["observations"]["storage"] == "temporary")
    return execute_pending(store, prepared, broker, settings, limits, saved["seed"], observer, retry_failures)


def execute_pending(store, prepared, broker, settings, limits, seed, observer, retry_failures=False):
    started = time.monotonic()
    with store.writer():
        latest = {r["task_id"]: r for r in store.attempts()}
        eligible = {t.task_id: t for t in prepared.tasks()}
        for raw_task in store.selection:
            task = eligible[raw_task["task_id"]]
            if task.model_dump(mode="json") != raw_task:
                raise ValueError("saved solver input differs from prepared task; create a new run")
            previous = latest.get(task.task_id)
            if previous and previous["status"] != "interrupted":
                failed = previous["status"] in {"error", "model_failure", "invalid_output", "evaluation_error", "time_limit", "call_limit", "tool_limit"}
                if not (retry_failures and failed):
                    observer({"kind": "task_skipped", "task_id": task.task_id, "reason": "already committed", "attempt_id": previous["attempt_id"]})
                    continue
            if time.monotonic() - started >= limits.run_timeout_s:
                observer({"kind": "run_time_limit", "message": "Remaining tasks are pending; resume this run."})
                break
            attempt = store.start(task.task_id)
            remaining = limits.run_timeout_s - (time.monotonic() - started)
            runner = SessionRunner(broker, settings,
                                   replace(limits, task_timeout_s=min(limits.task_timeout_s, remaining)), seed, observer)
            try:
                with tempfile.TemporaryDirectory(prefix="blackboard-task-") as directory:
                    input_dir = Path(directory) / "inputs"
                    prepared.copy_inputs(task, input_dir)
                    result = runner.execute(task, input_dir, attempt)
                output = attempt.freeze(result.pop("output"))
                result["output"] = output
                try:
                    result["grade"] = prepared.evaluate(task.task_id, output)
                except (ValueError, KeyError, TypeError) as exc:
                    result["execution_status"] = result["status"]
                    result["status"] = "evaluation_error"
                    result["grade"] = {"status": "error", "correct": None, "score": None, "error": str(exc)}
                attempt.event("evaluation", grade=result["grade"])
                committed = attempt.finish(result)
            except OSError as exc:
                # Disk/Drive/ledger failures must not turn into a model error or launch
                # another task. The partial attempt remains available for inspection.
                raise PersistenceError(f"Storage or task-input I/O failed; batch stopped: {exc}") from exc
            observer({"kind": "attempt_saved", "task_id": task.task_id, "attempt_id": committed["attempt_id"],
                      "status": committed["status"], "grade": committed.get("grade"), "run_dir": str(store.root)})
            store.summarize()
            if result["status"] == "interrupted":
                break
        summary = store.summarize()
    return store, summary
