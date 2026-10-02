"""
ingest/base.py
Base abstractions and utilities for benchmark dataset ingestion adapters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union

from contracts.schemas import BenchmarkTask, BlackboardState, TrialConfig

TaskT = TypeVar("TaskT", bound=BenchmarkTask)


class IngestionError(Exception):
    """Exception raised when benchmark dataset ingestion, parsing, or formatting fails."""
    pass


class BaseBenchmarkAdapter(ABC, Generic[TaskT]):
    """
    Abstract base adapter for parsing, normalizing, and formatting benchmark tasks
    into the Intelligible Blackboard Architecture's Pydantic contracts.
    """

    task_cls: Type[TaskT]

    @abstractmethod
    def parse_task(self, data: Dict[str, Any]) -> TaskT:
        """Parse raw task dictionary into typed BenchmarkTask model."""
        pass

    def load_from_dict(self, data: Dict[str, Any]) -> TaskT:
        """
        Load and validate a single task from dictionary.
        
        Args:
            data: Dictionary representation of a task item.
            
        Returns:
            Validated task instance of type TaskT.
            
        Raises:
            IngestionError: If input is not a dict or validation fails.
        """
        if not isinstance(data, dict):
            raise IngestionError(f"Expected dict input, got {type(data).__name__}")
        try:
            return self.parse_task(data)
        except IngestionError:
            raise
        except Exception as e:
            raise IngestionError(f"Failed to parse task: {e}") from e

    def load_from_json(self, json_str: str) -> Union[TaskT, List[TaskT]]:
        """
        Load task(s) from a JSON string representation.
        
        Args:
            json_str: Raw JSON string encoding a single task object or list of tasks.
            
        Returns:
            Single TaskT instance if JSON is an object, or List[TaskT] if JSON is an array.
            
        Raises:
            IngestionError: If JSON decoding fails or contents are malformed.
        """
        if not isinstance(json_str, str) or not json_str.strip():
            raise IngestionError("JSON string must not be empty or non-string.")

        try:
            parsed = json.loads(json_str)
        except Exception as e:
            raise IngestionError(f"Invalid JSON content: {e}") from e

        if isinstance(parsed, list):
            return [self.load_from_dict(item) for item in parsed]
        elif isinstance(parsed, dict):
            # Check if this is a dictionary with a tasks/data container list
            if "tasks" in parsed and isinstance(parsed["tasks"], list):
                return [self.load_from_dict(item) for item in parsed["tasks"]]
            if "data" in parsed and isinstance(parsed["data"], list):
                return [self.load_from_dict(item) for item in parsed["data"]]
            return self.load_from_dict(parsed)
        else:
            raise IngestionError(f"Expected dict or list in JSON, got {type(parsed).__name__}")

    def load_from_file(self, file_path: Union[str, Path]) -> List[TaskT]:
        """
        Load tasks from a .json or .jsonl file.
        
        Args:
            file_path: Path to target file.
            
        Returns:
            List of parsed TaskT instances.
            
        Raises:
            FileNotFoundError: If the target file does not exist.
            IngestionError: If file parsing fails or content is invalid.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise IngestionError(f"Expected a file path, got directory: {path}")

        tasks: List[TaskT] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                if path.suffix.lower() == ".jsonl":
                    for line_idx, line in enumerate(f, start=1):
                        line_str = line.strip()
                        if not line_str:
                            continue
                        try:
                            item = json.loads(line_str)
                            tasks.append(self.load_from_dict(item))
                        except IngestionError as ie:
                            raise IngestionError(f"Line {line_idx} in {path.name}: {ie}") from ie
                        except Exception as e:
                            raise IngestionError(f"Line {line_idx} in {path.name} is invalid JSON: {e}") from e
                else:
                    content = json.load(f)
                    if isinstance(content, list):
                        for idx, item in enumerate(content):
                            try:
                                tasks.append(self.load_from_dict(item))
                            except IngestionError as ie:
                                raise IngestionError(f"Item {idx} in {path.name}: {ie}") from ie
                    elif isinstance(content, dict):
                        if "tasks" in content and isinstance(content["tasks"], list):
                            for idx, item in enumerate(content["tasks"]):
                                tasks.append(self.load_from_dict(item))
                        elif "data" in content and isinstance(content["data"], list):
                            for idx, item in enumerate(content["data"]):
                                tasks.append(self.load_from_dict(item))
                        else:
                            tasks.append(self.load_from_dict(content))
                    else:
                        raise IngestionError(f"Unsupported JSON root type: {type(content).__name__}")
        except (FileNotFoundError, IngestionError):
            raise
        except Exception as e:
            raise IngestionError(f"Failed to read file {path}: {e}") from e

        return tasks

    def load_dataset(self, source_path: Union[str, Path], pattern: str = "*.json") -> List[TaskT]:
        """
        Load tasks from a single file or an entire directory matching pattern.
        
        Args:
            source_path: Path to file or directory.
            pattern: Glob pattern when source_path is a directory (default: '*.json').
            
        Returns:
            Aggregated list of all parsed TaskT instances.
        """
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset path not found: {path}")

        if path.is_file():
            return self.load_from_file(path)

        all_tasks: List[TaskT] = []
        matching_files = sorted(path.glob(pattern))
        for file in matching_files:
            if file.is_file():
                all_tasks.extend(self.load_from_file(file))
        return all_tasks

    def format_for_blackboard(
        self,
        task: Union[TaskT, Dict[str, Any]],
        session_id: Optional[str] = None,
    ) -> BlackboardState:
        """
        Format a benchmark task into an initialized BlackboardState contract.
        
        Args:
            task: Pre-parsed TaskT or raw task dictionary.
            session_id: Optional custom session ID.
            
        Returns:
            BlackboardState ready for Blackboard repository or Scheduler initialization.
        """
        if isinstance(task, dict):
            task_obj = self.load_from_dict(task)
        else:
            task_obj = task
        return task_obj.to_blackboard_state(session_id=session_id)

    def format_for_trial(
        self,
        task: Union[TaskT, Dict[str, Any]],
        counterfactual_density: float = 0.0,
        agent_models: Optional[Dict[str, str]] = None,
        max_turns: int = 15,
        seed: int = 42,
    ) -> TrialConfig:
        """
        Format a benchmark task into a TrialConfig contract for ablation experiments.
        
        Args:
            task: Pre-parsed TaskT or raw task dictionary.
            counterfactual_density: Fraction of counterfactual agents in ensemble [0.0, 1.0].
            agent_models: Optional role-to-model mapping.
            max_turns: Maximum conversation turns before timeout.
            seed: Random seed for reproducibility.
            
        Returns:
            TrialConfig instance conforming to contracts.schemas.TrialConfig.
        """
        if isinstance(task, dict):
            task_obj = self.load_from_dict(task)
        else:
            task_obj = task
        return task_obj.to_trial_config(
            counterfactual_density=counterfactual_density,
            agent_models=agent_models,
            max_turns=max_turns,
            seed=seed,
        )
