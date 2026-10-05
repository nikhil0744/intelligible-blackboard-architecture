"""
benchmarks/evaluator.py
Unified benchmark evaluator and diagnostic verification suite.

Evaluates multi-agent blackboard predictions against benchmark reference solutions,
and runs diagnostic checks ensuring source task IDs, answer types, and constraints
are preserved while stripping reference solutions from agent-visible inputs.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, ConfigDict, Field

from contracts.schemas import BenchmarkTask, BenchmarkType, BlackboardState
from ingest.base import SOLUTION_KEYS


class EvaluationResult(BaseModel):
    """Structured evaluation outcome of an agent deliberation against ground truth."""
    model_config = ConfigDict(extra="ignore")

    task_id: str = Field(..., description="Unique source task identifier.")
    benchmark_type: BenchmarkType = Field(..., description="Benchmark identifier.")
    ground_truth: str = Field(..., description="Target reference solution or diagnosis.")
    predicted_claim: str = Field(..., description="Final claim produced by the multi-agent deliberation.")
    exact_match: bool = Field(..., description="Whether the prediction matches ground truth exactly.")
    normalized_match: bool = Field(..., description="Case-insensitive, whitespace-normalized match.")
    token_f1: float = Field(..., description="Token-level harmonic mean overlap score in [0.0, 1.0].")
    matched_differential: Optional[str] = Field(
        default=None,
        description="Matched differential diagnosis if applicable.",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic notes and context.")


class DiagnosticCheckResult(BaseModel):
    """Result of diagnostic checks verifying adapter safety and contract preservation."""
    model_config = ConfigDict(extra="ignore")

    task_id: str
    benchmark_type: BenchmarkType
    passed: bool
    source_task_id_preserved: bool
    answer_types_preserved: bool
    constraints_preserved: bool
    reference_solution_stripped_from_agent_input: bool
    details: Dict[str, Any] = Field(default_factory=dict)


def _tokenize(text: str) -> List[str]:
    """Tokenize string into alphanumeric tokens."""
    return re.findall(r"\w+", (text or "").lower())


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level precision, recall, and F1 between prediction and ground truth."""
    pred_tokens = _tokenize(prediction)
    gt_tokens = _tokenize(ground_truth)
    if not pred_tokens or not gt_tokens:
        return 1.0 if pred_tokens == gt_tokens else 0.0

    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    if precision + recall == 0.0:
        return 0.0
    return round((2.0 * precision * recall) / (precision + recall), 4)


class BenchmarkEvaluator:
    """Evaluates task outcomes and executes safety/integrity diagnostic checks."""

    @staticmethod
    def evaluate(task: BenchmarkTask, predicted_claim: str) -> EvaluationResult:
        """
        Evaluate a predicted claim against the task's ground truth.
        """
        gt_raw = str(task.ground_truth if task.ground_truth is not None else "").strip()
        pred_raw = str(predicted_claim if predicted_claim is not None else "").strip()

        # Normalization
        norm_gt = re.sub(r"[\s\-_,.:;]+", " ", gt_raw.lower()).strip()
        norm_pred = re.sub(r"[\s\-_,.:;]+", " ", pred_raw.lower()).strip()

        exact_match = bool(gt_raw and pred_raw) and (gt_raw == pred_raw)
        # Retain the historical diagnostic matching for non-empty text; prototype grading
        # uses a separate strict evaluator and never treats substrings as correct answers.
        normalized_match = bool(norm_gt and norm_pred) and (
            (norm_gt == norm_pred) or (norm_gt in norm_pred) or (norm_pred in norm_gt)
        )
        token_f1 = compute_token_f1(pred_raw, gt_raw)

        matched_diff: Optional[str] = None
        # Check against clinical differentials if present
        if hasattr(task, "differential_diagnoses") and getattr(task, "differential_diagnoses"):
            differentials = getattr(task, "differential_diagnoses")
            for diff in differentials:
                clean_diff = re.sub(r"[\s\-_,.:;]+", " ", str(diff).lower()).strip()
                if clean_diff and norm_pred and (clean_diff in norm_pred or norm_pred in clean_diff):
                    matched_diff = str(diff)
                    break

        return EvaluationResult(
            task_id=task.task_id,
            benchmark_type=task.benchmark_type,
            ground_truth=gt_raw,
            predicted_claim=pred_raw,
            exact_match=exact_match,
            normalized_match=normalized_match,
            token_f1=token_f1,
            matched_differential=matched_diff,
            metadata={"domain": getattr(task, "domain", "general")},
        )

    @staticmethod
    def verify_adapter_integrity(
        raw_source_dict: Dict[str, Any],
        parsed_task: BenchmarkTask,
        blackboard_state: BlackboardState,
    ) -> DiagnosticCheckResult:
        """
        Diagnostic check verifying that:
        1. Source task ID is preserved.
        2. Answer types / domain / expected structures are preserved.
        3. Constraints (subtasks, differential diagnoses, data sources) are preserved.
        4. Reference solutions are strictly stripped from agent-visible initial_context.
        """
        details: Dict[str, Any] = {}

        # 1. Source Task ID check
        expected_id = str(
            raw_source_dict.get("task_id")
            or raw_source_dict.get("id")
            or raw_source_dict.get("case_id")
            or raw_source_dict.get("workload_id")
            or raw_source_dict.get("eval_MRN")
            or ""
        ).strip()
        task_id_preserved = (parsed_task.task_id == expected_id) and (blackboard_state.task_id == expected_id)
        details["expected_id"] = expected_id
        details["parsed_id"] = parsed_task.task_id

        # 2. Answer types & domain preserved
        answer_types_preserved = True
        if hasattr(parsed_task, "domain") and getattr(parsed_task, "domain"):
            details["domain"] = getattr(parsed_task, "domain")
        if parsed_task.ground_truth is None and ("ground_truth" in raw_source_dict or "solution" in raw_source_dict or "sol" in raw_source_dict):
            answer_types_preserved = False
        details["ground_truth_present_in_task"] = bool(parsed_task.ground_truth)

        # 3. Constraints preserved
        constraints_preserved = True
        if "data_sources" in raw_source_dict and hasattr(parsed_task, "data_sources"):
            expected_sources = len(raw_source_dict["data_sources"])
            actual_sources = len(getattr(parsed_task, "data_sources"))
            if expected_sources != actual_sources:
                constraints_preserved = False
        if "differential_diagnoses" in raw_source_dict and hasattr(parsed_task, "differential_diagnoses"):
            expected_diffs = len(raw_source_dict["differential_diagnoses"])
            actual_diffs = len(getattr(parsed_task, "differential_diagnoses"))
            if expected_diffs != actual_diffs:
                constraints_preserved = False

        # 4. Reference solution stripping check
        # Recursively check agent-visible initial_context for any solution keys
        def _contains_solution_key(obj: Any) -> bool:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k.lower() in SOLUTION_KEYS:
                        return True
                    if _contains_solution_key(v):
                        return True
            elif isinstance(obj, list):
                for item in obj:
                    if _contains_solution_key(item):
                        return True
            return False

        has_solution_in_agent_context = _contains_solution_key(blackboard_state.initial_context)
        solution_stripped = not has_solution_in_agent_context
        details["has_solution_in_agent_context"] = has_solution_in_agent_context

        passed = (
            task_id_preserved
            and answer_types_preserved
            and constraints_preserved
            and solution_stripped
        )

        return DiagnosticCheckResult(
            task_id=parsed_task.task_id,
            benchmark_type=parsed_task.benchmark_type,
            passed=passed,
            source_task_id_preserved=task_id_preserved,
            answer_types_preserved=answer_types_preserved,
            constraints_preserved=constraints_preserved,
            reference_solution_stripped_from_agent_input=solution_stripped,
            details=details,
        )
