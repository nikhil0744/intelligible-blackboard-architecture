"""Pinned, checksum-verified benchmark preparation; reference data never enters TaskInput."""
from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import shutil
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


def resources():
    return json.loads(files("benchmarks").joinpath("resources.json").read_text())


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class TaskInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    files: tuple[str, ...]
    answer_type: str


def download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read(4 * 1024 * 1024 + 1)
    if len(data) > 4 * 1024 * 1024:
        raise ValueError("asset exceeds the selected profile's download limit")
    return data


def cache_path(cache: Path, benchmark="kramabench") -> Path:
    spec = resources()[benchmark]
    if not spec.get("available", True):
        raise ValueError(spec["reason"])
    return Path(cache) / benchmark / spec["revision"] / spec["profile"]


def verify(root: Path, spec: dict) -> None:
    for asset in spec["assets"]:
        p = root / asset["path"]
        if p.is_symlink() or not p.is_file() or p.stat().st_size != asset["bytes"] or digest(p) != asset["sha256"]:
            raise ValueError(f"Incomplete/corrupt cache asset {asset['path']}; remove this profile cache and prepare again")


def prepare(cache: Path, benchmark="kramabench", fetch=download) -> Path:
    spec = resources()[benchmark]
    root = cache_path(cache, benchmark)
    if root.exists():
        verify(root, spec)
        if json.loads((root / "ready.json").read_text())["resources"] != fingerprint(spec):
            raise ValueError("cache manifest does not match pinned resources")
        return root
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".preparing-", dir=root.parent) as directory:
        staging = Path(directory) / "assets"
        staging.mkdir()
        for asset in spec["assets"]:
            url = f"https://raw.githubusercontent.com/{spec['repo']}/{spec['revision']}/" + urllib.parse.quote(asset["path"])
            data = fetch(url)
            if len(data) != asset["bytes"] or hashlib.sha256(data).hexdigest() != asset["sha256"]:
                raise ValueError(f"Upstream asset verification failed: {asset['path']}")
            p = staging / asset["path"]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        verify(staging, spec)
        (staging / "ready.json").write_text(json.dumps({"resources": fingerprint(spec)}))
        os.rename(staging, root)
    return root


@dataclass
class PreparedBenchmark:
    root: Path
    spec: dict

    @classmethod
    def open(cls, cache: Path, benchmark="kramabench"):
        root = cache_path(cache, benchmark)
        spec = resources()[benchmark]
        verify(root, spec)
        if json.loads((root / "ready.json").read_text())["resources"] != fingerprint(spec):
            raise ValueError("incompatible cache")
        return cls(root, spec)

    def _records(self):
        return {str(t["id"]): t for t in json.loads((self.root / self.spec["workload"]).read_text())}

    def tasks(self) -> list[TaskInput]:
        raw = self._records()
        available_files = [Path(a["path"]).name for a in self.spec["assets"] if a["path"].endswith(".csv")]
        # Explicit allowlist: no labels, solution steps, subtasks, or raw metadata.
        tasks = []
        for tid in self.spec["task_ids"]:
            record = raw[tid]
            if record["answer_type"] != "numeric_exact" or not record["data_sources"] or not set(record["data_sources"]) <= set(available_files):
                raise ValueError(f"Task {tid} is incompatible with the pinned numeric CSV profile")
            tasks.append(TaskInput(task_id=tid, question=record["query"],
                                   files=tuple(record["data_sources"]), answer_type=record["answer_type"]))
        return tasks

    def coverage(self):
        records = self._records()
        return {"profile": self.spec["profile"], "selected_task_ids": self.spec["task_ids"],
                "total_domain_tasks": len(records), "eligible_tasks": len(self.spec["task_ids"]),
                "excluded_task_ids": sorted(set(records) - set(self.spec["task_ids"])),
                "metrics": ["success"], "unavailable_metrics": ["pipeline", "hosted_judge", "subtasks"],
                "note": self.spec["coverage_note"]}

    def copy_inputs(self, task: TaskInput, target: Path):
        target.mkdir(parents=True, exist_ok=False)
        assets = {Path(a["path"]).name: a["path"] for a in self.spec["assets"] if a["path"].endswith(".csv")}
        for name in task.files:
            shutil.copyfile(self.root / assets[name], target / name)

    def evaluate(self, task_id: str, output):
        if task_id not in self.spec["task_ids"]:
            raise ValueError("task is outside the declared grading profile")
        if output is None:
            return {"metric": "success", "score": 0, "correct": False, "status": "no_final_submission"}
        # Execute only three audited, checksum-pinned definitions; importing the upstream
        # module would initialize hosted-model grading paths. No solver text is executed.
        metric_path = self.root / "benchmark/metrics.py"
        asset = next(a for a in self.spec["assets"] if a["path"] == "benchmark/metrics.py")
        if digest(metric_path) != asset["sha256"]:
            raise ValueError("evaluator checksum changed")
        tree = ast.parse(metric_path.read_text())
        names = {"str_to_float", "try_convert_to_number", "Success"}
        definitions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        if {n.name for n in definitions} != names:
            raise ValueError("pinned metric definitions missing")
        namespace = {"Metric": object, "logging": logging}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(metric_path), "exec"), namespace)
        reference = self._records()[task_id]["answer"]  # accessed only after output is frozen
        score = namespace["Success"]()(output, reference)[0]
        return {"metric": "success", "score": score, "correct": bool(score), "status": "scored",
                "evaluator_revision": self.spec["revision"], "official_aggregate": False}
