"""
benchmarks/environment.py
Benchmark execution environments for KramaBench and MedAgentBench.

Implements Student 1's benchmark environment requirements:
1. Isolated workspaces with path safety and clean directory lifecycles.
2. Controlled action registries enforcing strict tool access.
3. Enforced execution timeouts on tool actions.
4. Structured observation capture with timing and telemetry.
5. Clean environment resets between task/seed cases.
6. Seamless coordination with Student 3's SandboxSession specifications
   so tool states and database writes/generated files remain isolated and
   never leak across trials or affect live environments.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import concurrent.futures
import copy
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from contracts.schemas import (
    BenchmarkTask,
    BenchmarkType,
    BlackboardState,
    BlackboardStatus,
)
from ingest.base import strip_reference_solutions
from ingest.kramabench import KramaBenchTask
from ingest.medagentbench import MedAgentBenchTask

logger = logging.getLogger("benchmarks.environment")


class ActionRequest(BaseModel):
    """Normalized specification for a tool/action request within a benchmark environment."""
    model_config = ConfigDict(extra="ignore")

    action: str = Field(..., description="Action name registered in environment whitelist.")
    params: Dict[str, Any] = Field(default_factory=dict, description="Action arguments.")
    timeout_seconds: Optional[float] = Field(
        default=None,
        description="Optional timeout override in seconds for this specific action.",
    )
    calling_agent_id: Optional[str] = Field(
        default=None,
        description="Identifier of the agent invoking the action.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Target session ID.",
    )


class ActionObservation(BaseModel):
    """Captured observation and execution telemetry from a controlled tool invocation."""
    model_config = ConfigDict(extra="ignore")

    action: str = Field(..., description="Action name that was invoked.")
    success: bool = Field(..., description="Whether the action succeeded without errors.")
    data: Any = Field(default=None, description="Observation payload or structured result.")
    error: Optional[str] = Field(default=None, description="Error message if action failed or timed out.")
    execution_time_ms: float = Field(default=0.0, description="Elapsed execution time in milliseconds.")
    timed_out: bool = Field(default=False, description="Whether execution was terminated due to timeout.")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the observation.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Supplemental telemetry or environmental context.",
    )


class BaseBenchmarkEnvironment(ABC):
    """
    Abstract base environment providing isolated workspaces, controlled actions,
    execution timeouts, observation capture, and sandbox branch coordination.
    """

    def __init__(
        self,
        task: BenchmarkTask,
        session_id: Optional[str] = None,
        branch_id: str = "main",
        workspace_base_dir: Optional[Union[str, Path]] = None,
        default_timeout_seconds: float = 5.0,
        seed: Optional[int] = 42,
    ):
        self.task = task
        self.session_id = session_id or f"env_{task.benchmark_type.value}_{task.task_id}_{uuid4().hex[:8]}"
        self.branch_id = branch_id
        self.default_timeout_seconds = default_timeout_seconds
        self.seed = seed
        self._lock = threading.RLock()
        self._is_closed = False
        self._observation_history: List[ActionObservation] = []

        # Setup isolated workspace directory
        if workspace_base_dir:
            base = Path(workspace_base_dir)
            base.mkdir(parents=True, exist_ok=True)
            self._temp_dir: Optional[tempfile.TemporaryDirectory] = None
            self.workspace_dir = base / f"{self.session_id}_{self.branch_id}"
            self.workspace_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._temp_dir = tempfile.TemporaryDirectory(prefix=f"bb_ws_{self.branch_id}_")
            self.workspace_dir = Path(self._temp_dir.name)

        # Threadpool for running actions with hard execution timeouts
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    @property
    def is_closed(self) -> bool:
        return self._is_closed

    @abstractmethod
    def get_permitted_actions(self) -> Set[str]:
        """Return the strict whitelist of permitted tool/action names for this environment."""
        pass

    @abstractmethod
    def _dispatch_action(self, action: str, params: Dict[str, Any]) -> Any:
        """Internal synchronous execution of the validated action."""
        pass

    def _resolve_safe_path(self, relative_path: Union[str, Path]) -> Path:
        """
        Validate and resolve a path within the isolated workspace.
        Prevents directory traversal attacks (e.g. '../').
        """
        target = (self.workspace_dir / relative_path).resolve()
        workspace_resolved = self.workspace_dir.resolve()
        if not str(target).startswith(str(workspace_resolved)):
            raise PermissionError(f"Path traversal denied: '{relative_path}' escapes isolated workspace.")
        return target

    def execute_action(
        self,
        action: str,
        params: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[float] = None,
        calling_agent_id: Optional[str] = None,
    ) -> ActionObservation:
        """
        Execute a controlled tool action within the isolated workspace with timeout enforcement.
        """
        with self._lock:
            if self._is_closed:
                return ActionObservation(
                    action=action,
                    success=False,
                    error="Environment is closed.",
                    execution_time_ms=0.0,
                )

            permitted = self.get_permitted_actions()
            if action not in permitted:
                obs = ActionObservation(
                    action=action,
                    success=False,
                    error=f"Unauthorized action '{action}'. Permitted actions: {sorted(list(permitted))}",
                    execution_time_ms=0.0,
                )
                self._observation_history.append(obs)
                return obs

            actual_params = params or {}
            timeout = timeout_seconds if timeout_seconds is not None else self.default_timeout_seconds
            t0 = time.perf_counter()

            # Execute with timeout in worker thread
            future = self._executor.submit(self._dispatch_action, action, actual_params)
            try:
                result = future.result(timeout=timeout)
                duration_ms = (time.perf_counter() - t0) * 1000.0
                obs = ActionObservation(
                    action=action,
                    success=True,
                    data=result,
                    execution_time_ms=round(duration_ms, 2),
                    metadata={"branch_id": self.branch_id, "agent_id": calling_agent_id},
                )
            except concurrent.futures.TimeoutError:
                duration_ms = (time.perf_counter() - t0) * 1000.0
                obs = ActionObservation(
                    action=action,
                    success=False,
                    error=f"Action '{action}' timed out after {timeout:.2f}s",
                    execution_time_ms=round(duration_ms, 2),
                    timed_out=True,
                    metadata={"branch_id": self.branch_id, "agent_id": calling_agent_id},
                )
            except Exception as e:
                duration_ms = (time.perf_counter() - t0) * 1000.0
                obs = ActionObservation(
                    action=action,
                    success=False,
                    error=f"Action execution error: {type(e).__name__}: {str(e)}",
                    execution_time_ms=round(duration_ms, 2),
                    metadata={"branch_id": self.branch_id, "agent_id": calling_agent_id},
                )

            self._observation_history.append(obs)
            return obs

    def get_observation_history(self) -> List[ActionObservation]:
        """Return a copy of the captured observation history."""
        with self._lock:
            return [obs.model_copy() for obs in self._observation_history]

    def reset(
        self,
        task: Optional[BenchmarkTask] = None,
        seed: Optional[int] = None,
    ) -> BlackboardState:
        """
        Cleanly reset the environment:
        1. Purge all generated files in the workspace directory.
        2. Clear observation history and internal tool state.
        3. Update task and seed if provided.
        4. Return a sanitized BlackboardState with reference solutions stripped.
        """
        with self._lock:
            if task is not None:
                self.task = task
            if seed is not None:
                self.seed = seed

            self._observation_history.clear()

            # Clean workspace contents
            if self.workspace_dir.exists():
                for item in self.workspace_dir.iterdir():
                    try:
                        if item.is_dir():
                            shutil.rmtree(item, ignore_errors=True)
                        else:
                            item.unlink(missing_ok=True)
                    except Exception as e:
                        logger.warning("Error cleaning workspace item %s: %s", item, e)

            # Produce clean, sanitized initial state
            raw_state = self.task.to_blackboard_state(session_id=self.session_id)
            clean_context = strip_reference_solutions(raw_state.initial_context)

            return BlackboardState(
                session_id=self.session_id,
                task_id=self.task.task_id,
                task_description=raw_state.task_description,
                ground_truth=self.task.ground_truth,
                initial_context=clean_context,
                status=BlackboardStatus.ACTIVE,
                entries=[],
                active_branches=[self.branch_id],
                metadata=dict(raw_state.metadata),
            )

    @abstractmethod
    def fork_sandbox_environment(self, sandbox_branch_id: str) -> "BaseBenchmarkEnvironment":
        """
        Coordinate with Student 3's SandboxSession specifications:
        Fork an isolated branch environment with copy-on-write or deep-copied
        workspace files and tool states. Any writes or modifications within the
        sandbox branch must NEVER leak back to the parent environment or live board.
        """
        pass

    def close(self) -> None:
        """Cleanly tear down the isolated workspace and terminate threadpools."""
        with self._lock:
            if self._is_closed:
                return
            self._is_closed = True
            self._executor.shutdown(wait=False, cancel_futures=True)
            if self._temp_dir is not None:
                try:
                    self._temp_dir.cleanup()
                except Exception as e:
                    logger.warning("Temporary workspace cleanup error: %s", e)
            elif self.workspace_dir.exists():
                try:
                    shutil.rmtree(self.workspace_dir, ignore_errors=True)
                except Exception as e:
                    logger.warning("Workspace directory cleanup error: %s", e)

    def __enter__(self) -> "BaseBenchmarkEnvironment":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class KramaBenchEnvironment(BaseBenchmarkEnvironment):
    """
    Isolated execution and tool environment for KramaBench multi-agent data lake exploration.
    Supports controlled data lake search, schema inspection, aggregations, and isolated notes.
    """

    PERMITTED_ACTIONS: Set[str] = {
        "list_data_sources",
        "inspect_schema",
        "search_records",
        "compute_correlation",
        "write_intermediate_note",
        "read_intermediate_notes",
    }

    def __init__(
        self,
        task: KramaBenchTask,
        session_id: Optional[str] = None,
        branch_id: str = "main",
        workspace_base_dir: Optional[Union[str, Path]] = None,
        default_timeout_seconds: float = 5.0,
        seed: Optional[int] = 42,
    ):
        super().__init__(
            task=task,
            session_id=session_id,
            branch_id=branch_id,
            workspace_base_dir=workspace_base_dir,
            default_timeout_seconds=default_timeout_seconds,
            seed=seed,
        )
        self.krama_task: KramaBenchTask = task
        # Seed simulated data lake files if not already present
        self._init_data_lake_files()

    def get_permitted_actions(self) -> Set[str]:
        return set(self.PERMITTED_ACTIONS)

    def _init_data_lake_files(self) -> None:
        """Materialize dummy/mock data lake files in the workspace for inspection."""
        for src in self.krama_task.data_sources:
            safe_file = self._resolve_safe_path(Path(src).name)
            if not safe_file.exists():
                meta_info = {
                    "source_name": src,
                    "domain": self.krama_task.domain,
                    "simulated_row_count": 500,
                    "columns": ["timestamp", "value", "sensor_id", "region_code", "status"],
                }
                safe_file.write_text(json.dumps(meta_info, indent=2), encoding="utf-8")

    def _dispatch_action(self, action: str, params: Dict[str, Any]) -> Any:
        if action == "list_data_sources":
            return {
                "task_id": self.krama_task.task_id,
                "domain": self.krama_task.domain,
                "data_sources": list(self.krama_task.data_sources),
                "workspace_files": [f.name for f in self.workspace_dir.iterdir() if f.is_file()],
            }

        elif action == "inspect_schema":
            source_name = Path(str(params.get("source", ""))).name
            if not source_name:
                raise ValueError("Parameter 'source' is required for inspect_schema.")
            safe_file = self._resolve_safe_path(source_name)
            if safe_file.exists():
                try:
                    return json.loads(safe_file.read_text(encoding="utf-8"))
                except Exception:
                    pass
            # Fallback schema descriptor
            return {
                "source": source_name,
                "domain": self.krama_task.domain,
                "columns": ["id", "timestamp", "observation_value", "category", "confidence"],
                "format": safe_file.suffix.replace(".", "") or "csv",
            }

        elif action == "search_records":
            query = str(params.get("query", "")).strip()
            limit = int(params.get("limit", 5))
            source = str(params.get("source", "all"))
            return {
                "matched_count": limit,
                "source": source,
                "query": query,
                "records": [
                    {
                        "record_id": f"rec_{i+1}",
                        "source": source,
                        "observation": f"Finding related to '{query}' in {self.krama_task.domain}",
                        "signal_strength": round(0.70 + (i * 0.05), 3),
                    }
                    for i in range(min(limit, 10))
                ],
            }

        elif action == "compute_correlation":
            var1 = str(params.get("var1", "x"))
            var2 = str(params.get("var2", "y"))
            # Deterministic calculation based on variable names and seed
            seed_offset = hash(f"{var1}_{var2}_{self.seed}") % 100
            val = -0.85 + (seed_offset * 0.015)
            val = max(-1.0, min(1.0, val))
            return {
                "var1": var1,
                "var2": var2,
                "metric": "pearson",
                "coefficient": round(val, 4),
                "p_value": 0.001,
                "confidence_interval": [round(val - 0.05, 3), round(val + 0.05, 3)],
            }

        elif action == "write_intermediate_note":
            title = Path(str(params.get("title", "note.txt"))).name
            content = str(params.get("content", ""))
            safe_target = self._resolve_safe_path(f"notes_{title}")
            safe_target.write_text(content, encoding="utf-8")
            return {"file": safe_target.name, "bytes_written": len(content)}

        elif action == "read_intermediate_notes":
            notes = {}
            for item in self.workspace_dir.glob("notes_*"):
                if item.is_file():
                    notes[item.name] = item.read_text(encoding="utf-8")
            return {"notes_count": len(notes), "notes": notes}

        raise ValueError(f"Unrecognized action handler: {action}")

    def fork_sandbox_environment(self, sandbox_branch_id: str) -> "KramaBenchEnvironment":
        """
        Coordinate with Student 3's SandboxSession specifications:
        Clone workspace files and state into an isolated branch environment.
        """
        with self._lock:
            child_ws = self.workspace_dir / "sandbox_branches" / sandbox_branch_id
            child_ws.mkdir(parents=True, exist_ok=True)

            # Copy existing workspace files to the sandbox branch
            for item in self.workspace_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, child_ws / item.name)

            child_env = KramaBenchEnvironment(
                task=copy.deepcopy(self.krama_task),
                session_id=f"{self.session_id}_{sandbox_branch_id}",
                branch_id=sandbox_branch_id,
                workspace_base_dir=child_ws.parent,
                default_timeout_seconds=self.default_timeout_seconds,
                seed=self.seed,
            )
            child_env._observation_history = [obs.model_copy() for obs in self._observation_history]
            return child_env


class MedAgentBenchEnvironment(BaseBenchmarkEnvironment):
    """
    Isolated clinical environment for MedAgentBench diagnostic consensus.
    Provides controlled access to EHR records, vitals, lab reports, imaging,
    and branch-isolated test orders without leaking ground-truth diagnosis.
    """

    PERMITTED_ACTIONS: Set[str] = {
        "get_patient_vitals",
        "get_lab_results",
        "get_imaging_reports",
        "get_clinical_history",
        "order_diagnostic_test",
        "get_ordered_tests",
        "consult_clinical_guidelines",
    }

    def __init__(
        self,
        task: MedAgentBenchTask,
        session_id: Optional[str] = None,
        branch_id: str = "main",
        workspace_base_dir: Optional[Union[str, Path]] = None,
        default_timeout_seconds: float = 5.0,
        seed: Optional[int] = 42,
    ):
        super().__init__(
            task=task,
            session_id=session_id,
            branch_id=branch_id,
            workspace_base_dir=workspace_base_dir,
            default_timeout_seconds=default_timeout_seconds,
            seed=seed,
        )
        self.med_task: MedAgentBenchTask = task
        self._ordered_tests: List[Dict[str, Any]] = []

    def get_permitted_actions(self) -> Set[str]:
        return set(self.PERMITTED_ACTIONS)

    def _dispatch_action(self, action: str, params: Dict[str, Any]) -> Any:
        profile = self.med_task.patient_profile or {}

        if action == "get_patient_vitals":
            vitals = profile.get("vitals") or {
                "blood_pressure": "142/88 mmHg",
                "heart_rate": "96 bpm",
                "respiratory_rate": "22 /min",
                "temperature": "37.2 C",
                "spo2": "93% on room air",
            }
            return {
                "patient_id": self.med_task.patient_id or "patient_01",
                "vitals": vitals,
            }

        elif action == "get_lab_results":
            category = str(params.get("category", "all")).lower()
            labs = profile.get("labs") or profile.get("laboratory") or {
                "cardiac": {"troponin_i": "0.04 ng/mL", "bnp": "480 pg/mL"},
                "hematology": {"wbc": "9.8 x10^3/uL", "hgb": "13.2 g/dL", "platelets": "210 x10^3/uL"},
                "metabolic": {"creatinine": "1.1 mg/dL", "egfr": "68 mL/min", "lactate": "1.6 mmol/L"},
            }
            if category in labs and isinstance(labs, dict):
                return {"category": category, "results": labs[category]}
            return {"category": "all", "results": labs}

        elif action == "get_imaging_reports":
            modality = str(params.get("modality", "all")).lower()
            imaging = profile.get("imaging") or profile.get("radiology") or {
                "cxr": "Cardiomegaly with bilateral perihilar haziness and trace pleural blunting.",
                "hrct": "Subpleural basilar reticular opacities without honeycomb change.",
                "echo": "LVEF 45% with mild diastolic dysfunction.",
            }
            if modality in imaging and isinstance(imaging, dict):
                return {"modality": modality, "report": imaging[modality]}
            return {"imaging_reports": imaging}

        elif action == "get_clinical_history":
            return {
                "patient_id": self.med_task.patient_id,
                "problem_statement": self.med_task.problem_statement,
                "differential_diagnoses": list(self.med_task.differential_diagnoses),
                "clinical_notes": profile.get("clinical_notes") or profile.get("history") or "No prior surgical interventions.",
            }

        elif action == "order_diagnostic_test":
            test_name = str(params.get("test_name", "")).strip()
            rationale = str(params.get("rationale", "")).strip()
            if not test_name:
                raise ValueError("Parameter 'test_name' is required.")

            order = {
                "order_id": f"order_{len(self._ordered_tests) + 1}",
                "test_name": test_name,
                "rationale": rationale,
                "ordered_at": datetime.now(timezone.utc).isoformat(),
                "branch_id": self.branch_id,
            }
            self._ordered_tests.append(order)
            return {"status": "ORDERED", "order": order}

        elif action == "get_ordered_tests":
            return {"ordered_tests_count": len(self._ordered_tests), "orders": list(self._ordered_tests)}

        elif action == "consult_clinical_guidelines":
            condition = str(params.get("condition", "")).strip()
            specialty = str(params.get("specialty", "general")).strip()
            return {
                "specialty": specialty,
                "condition": condition,
                "guideline": f"Consensus recommendation for {condition or 'presentation'}: Correlate biomarker trajectories with high-resolution imaging prior to definitive diagnostic assertion.",
            }

        raise ValueError(f"Unrecognized action handler: {action}")

    def fork_sandbox_environment(self, sandbox_branch_id: str) -> "MedAgentBenchEnvironment":
        """
        Coordinate with Student 3's SandboxSession specifications:
        Clone patient profile, order state, and observation history into
        an isolated sandbox branch environment.
        """
        with self._lock:
            child_ws = self.workspace_dir / "sandbox_branches" / sandbox_branch_id
            child_ws.mkdir(parents=True, exist_ok=True)

            child_env = MedAgentBenchEnvironment(
                task=copy.deepcopy(self.med_task),
                session_id=f"{self.session_id}_{sandbox_branch_id}",
                branch_id=sandbox_branch_id,
                workspace_base_dir=child_ws.parent,
                default_timeout_seconds=self.default_timeout_seconds,
                seed=self.seed,
            )
            # Deep-copy ordered tests and observations so sandbox branch is isolated
            child_env._ordered_tests = copy.deepcopy(self._ordered_tests)
            child_env._observation_history = [obs.model_copy() for obs in self._observation_history]
            return child_env

    def reset(
        self,
        task: Optional[BenchmarkTask] = None,
        seed: Optional[int] = None,
    ) -> BlackboardState:
        with self._lock:
            if isinstance(task, MedAgentBenchTask):
                self.med_task = task
            self._ordered_tests.clear()
            return super().reset(task=task, seed=seed)
