"""Pre-run checks for notebook / local inference (task S2.4).

    python -m scripts.s2_preflight            # prints a report, exit code 1 if a required check fails

Checks hardware (GPU, memory, disk), the backend's health, that the configured model is
present (it is never pulled or swapped automatically), a real probe call (usage reported,
context window honoured, cold-load time), and dependency versions. The report is plain
data so it can be stored in a run manifest.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import time
from importlib import metadata
from typing import Any, Callable, Dict, List, Optional

from .broker import ModelBroker
from .config import InferenceSettings
from .types import ChatMessage, LLMError, LLMRequest

DEPENDENCIES = ("pydantic", "redis", "fastapi", "pandas", "litellm", "blackboard-agents")


def _check(name: str, ok: bool, detail: Any = None, required: bool = True) -> Dict[str, Any]:
    return {"name": name, "ok": bool(ok), "required": required, "detail": detail}


def detect_gpus(run: Callable[..., Any] = subprocess.run) -> List[Dict[str, Any]]:
    """GPUs reported by nvidia-smi; [] when there is no NVIDIA GPU or driver."""
    try:
        out = run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            gpus.append({"name": parts[0], "memory_total_mb": int(float(parts[1])), "memory_free_mb": int(float(parts[2]))})
    return gpus


def system_memory_mb() -> Optional[Dict[str, int]]:
    try:
        info = {}
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                k, v = line.split(":", 1)
                info[k] = int(v.split()[0]) // 1024
        return {"total_mb": info["MemTotal"], "available_mb": info.get("MemAvailable", info.get("MemFree", 0))}
    except (OSError, KeyError, ValueError):
        return None  # not Linux: unknown, not zero


def dependency_versions() -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {"python": platform.python_version(), "platform": platform.platform()}
    for name in DEPENDENCIES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def hardware_report() -> Dict[str, Any]:
    disk = shutil.disk_usage(".")
    return {
        "gpus": detect_gpus(),
        "memory": system_memory_mb(),
        "disk_free_mb": disk.free // (1024 * 1024),
    }


def run_preflight(broker: ModelBroker, settings: Optional[InferenceSettings] = None, probe: bool = True) -> Dict[str, Any]:
    """Return {"ok", "checks", "hardware", "dependencies", "settings", "model", ...}. Never raises on a failed check."""
    settings = settings or getattr(broker, "settings", None) or InferenceSettings(model=broker.default_model)
    model = broker.default_model
    be = broker.backend
    checks: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {
        "backend": be.name,
        "model": model,
        "model_digest": None,
        "settings": settings.to_dict(),
        "hardware": hardware_report(),
        "dependencies": dependency_versions(),
        "checks": checks,
    }

    gpus = report["hardware"]["gpus"]
    checks.append(_check("gpu_available", bool(gpus), gpus or "no NVIDIA GPU detected (CPU inference will be slow)", required=False))
    py_ok = sys.version_info >= (3, 11)
    checks.append(_check("python_version", py_ok, platform.python_version(), required=False))

    if be.name == "ollama":
        try:
            report["backend_version"] = be.version()
            checks.append(_check("backend_reachable", True, f"Ollama {report['backend_version']} at {be.host}"))
        except LLMError as e:
            checks.append(_check("backend_reachable", False, f"{e} -> start it with `ollama serve`"))
            report["ok"] = False
            return report
        entries = {m["name"]: m for m in be.model_entries()}
        if model in entries:
            entry = entries[model]
            report["model_digest"] = entry.get("digest")
            report["model_details"] = entry.get("details")
            checks.append(_check("model_available", True, {"digest": entry.get("digest"), "size_bytes": entry.get("size")}))
        else:
            checks.append(_check("model_available", False,
                                 f"{model!r} is not pulled (have: {sorted(entries)}). Run `ollama pull {model}`. "
                                 "No other model is substituted."))
            report["ok"] = False
            return report
        trained_ctx = be.trained_context_length(model)
        report["model_context_length"] = trained_ctx
        checks.append(_check("context_window_supported", trained_ctx is None or settings.num_ctx <= trained_ctx,
                             {"num_ctx": settings.num_ctx, "model_context_length": trained_ctx}))
        checks.append(_check("num_ctx_matches_backend", be.num_ctx == settings.num_ctx,
                             {"backend_num_ctx": be.num_ctx, "settings_num_ctx": settings.num_ctx}))
    else:
        checks.append(_check("backend_reachable", True, f"{be.name} (no health endpoint checked)", required=False))

    if probe:
        req = LLMRequest(
            messages=[ChatMessage(role="user", content='Reply with the JSON object {"ok": true} and nothing else.')],
            temperature=settings.temperature, max_tokens=32, seed=0,
            agent_id="_preflight", metadata={"phase": "tool", "trial_id": "_preflight"},
        )
        t0 = time.perf_counter()
        try:
            resp = broker.generate(req)
            report["probe"] = {
                "seconds_including_model_load": round(time.perf_counter() - t0, 2),
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
                "text": resp.text[:200],
            }
            checks.append(_check("probe_call", True, report["probe"]))
            checks.append(_check("usage_reported", resp.usage_known,
                                 "backend reports token counts" if resp.usage_known
                                 else "backend did not report token counts; usage will be recorded as unknown",
                                 required=False))
        except LLMError as e:
            checks.append(_check("probe_call", False, str(e)))
        if be.name == "ollama":
            loaded = {m.get("name"): m for m in be.running_models()}
            m = loaded.get(model)
            if m:
                size, vram = m.get("size") or 0, m.get("size_vram") or 0
                report["loaded_model"] = {"size_bytes": size, "size_vram_bytes": vram, "context_length": m.get("context_length")}
                checks.append(_check("model_fully_on_gpu", size > 0 and vram >= size,
                                     f"{vram / max(size, 1):.0%} of the model is in GPU memory", required=False))
            others = sorted(n for n in loaded if n != model)
            checks.append(_check("single_loaded_model", not others, others or "only the study model is loaded", required=False))

    report["ok"] = all(c["ok"] for c in checks if c["required"])
    return report


def format_report(report: Dict[str, Any]) -> str:
    lines = [f"backend={report['backend']} model={report['model']} digest={report.get('model_digest')}"]
    for c in report["checks"]:
        mark = "PASS" if c["ok"] else ("FAIL" if c["required"] else "WARN")
        lines.append(f"[{mark}] {c['name']}: {c['detail']}")
    lines.append("READY" if report.get("ok") else "NOT READY")
    return "\n".join(lines)
