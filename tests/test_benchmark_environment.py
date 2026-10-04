"""
tests/test_benchmark_environment.py
Comprehensive test suite verifying Student 1's benchmark environment requirements:
1. Workspace isolation & path safety.
2. Controlled action registries for KramaBench & MedAgentBench.
3. Execution timeout enforcement.
4. Structured observation capture & timing telemetry.
5. Clean environment resets between task/seed cases.
6. Student 3 sandbox coordination (tool state and file isolation on branches).
7. Benchmark ingestion diagnostic checks (source task ID, answer types, constraints preserved, solutions stripped).
8. BenchmarkEvaluator accuracy and clinical differential diagnosis matching.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import pytest

from benchmarks.environment import (
    ActionObservation,
    ActionRequest,
    KramaBenchEnvironment,
    MedAgentBenchEnvironment,
)
from benchmarks.evaluator import BenchmarkEvaluator, compute_token_f1
from contracts.schemas import BenchmarkType, BlackboardStatus
from ingest.kramabench import KramaBenchAdapter, KramaBenchTask
from ingest.medagentbench import MedAgentBenchAdapter, MedAgentBenchTask
from ingest.base import strip_reference_solutions


@pytest.fixture
def sample_krama_task() -> KramaBenchTask:
    adapter = KramaBenchAdapter()
    return adapter.load_from_dict({
        "id": "krama_wildfire_101",
        "query": "Correlate fuel moisture and wildfire perimeter expansion in the Quebec 2023 season.",
        "domain": "Wildfire Prevention",
        "data_sources": ["fuel_moisture_grid.csv", "satellite_burn_perimeter.parquet"],
        "subtasks": ["Filter perimeter >= 100ha", "Calculate Pearson correlation"],
        "expected_output": {"correlation": -0.82, "p_val": 0.001},
    })


@pytest.fixture
def sample_medagent_task() -> MedAgentBenchTask:
    adapter = MedAgentBenchAdapter()
    return adapter.load_from_dict({
        "case_id": "case_cardio_202",
        "problem_statement": "64-year-old male presenting with acute dyspnea, orthopnea, bilateral lower extremity edema, and elevated BNP.",
        "ground_truth": "Acute Decompensated Heart Failure (ADHF)",
        "patient_id": "PT_64M_99",
        "clinical_specialties": ["Cardiologist", "Pulmonologist"],
        "differential_diagnoses": [
            "Acute Decompensated Heart Failure (ADHF)",
            "Idiopathic Pulmonary Fibrosis (IPF)",
            "Acute Pulmonary Embolism",
        ],
        "context": {
            "vitals": {"blood_pressure": "158/94 mmHg", "heart_rate": "104 bpm", "spo2": "90%"},
            "labs": {
                "cardiac": {"troponin_i": "0.06 ng/mL", "bnp": "1250 pg/mL"},
                "metabolic": {"creatinine": "1.3 mg/dL"},
            },
            "imaging": {"cxr": "Bilateral pleural effusions and pulmonary vascular congestion."},
        },
    })


# ============================================================================
# 1. Isolated Workspace & Path Safety Tests
# ============================================================================

def test_kramabench_workspace_isolation_and_file_safety(sample_krama_task: KramaBenchTask):
    """Verify KramaBench creates isolated workspace and blocks directory traversal."""
    with KramaBenchEnvironment(sample_krama_task, branch_id="main") as env:
        assert env.workspace_dir.exists()
        assert env.workspace_dir.is_dir()

        # Writing safe intermediate note
        write_obs = env.execute_action(
            "write_intermediate_note",
            {"title": "analysis.txt", "content": "Found correlation between sensor and perimeter."},
        )
        assert write_obs.success is True
        assert (env.workspace_dir / "notes_analysis.txt").exists()

        # Path traversal attempt outside workspace must fail safely
        bad_obs = env.execute_action(
            "write_intermediate_note",
            {"title": "../stolen.txt", "content": "Attempted breakout"},
        )
        assert (env.workspace_dir / "notes_stolen.txt").exists()
        # Verify the file was not written outside workspace
        assert not (env.workspace_dir.parent / "stolen.txt").exists()


def test_workspace_clean_deletion_on_close(sample_krama_task: KramaBenchTask):
    """Verify workspace temporary directory is completely purged on close."""
    env = KramaBenchEnvironment(sample_krama_task)
    ws_dir = env.workspace_dir
    assert ws_dir.exists()

    env.close()
    assert env.is_closed is True
    assert not ws_dir.exists()


# ============================================================================
# 2. Controlled Action Registry Tests
# ============================================================================

def test_kramabench_controlled_actions(sample_krama_task: KramaBenchTask):
    """Verify all whitelisted KramaBench tools execute cleanly."""
    with KramaBenchEnvironment(sample_krama_task) as env:
        # 1. list_data_sources
        obs1 = env.execute_action("list_data_sources")
        assert obs1.success is True
        assert "fuel_moisture_grid.csv" in obs1.data["data_sources"]

        # 2. inspect_schema
        obs2 = env.execute_action("inspect_schema", {"source": "fuel_moisture_grid.csv"})
        assert obs2.success is True
        assert "columns" in obs2.data

        # 3. search_records
        obs3 = env.execute_action("search_records", {"query": "perimeter growth", "limit": 3})
        assert obs3.success is True
        assert len(obs3.data["records"]) == 3

        # 4. compute_correlation
        obs4 = env.execute_action("compute_correlation", {"var1": "moisture", "var2": "perimeter_delta"})
        assert obs4.success is True
        assert -1.0 <= obs4.data["coefficient"] <= 1.0

        # 5. write and read notes
        env.execute_action("write_intermediate_note", {"title": "findings.txt", "content": "Hypothesis validated"})
        obs5 = env.execute_action("read_intermediate_notes")
        assert obs5.success is True
        assert obs5.data["notes_count"] >= 1


def test_medagentbench_controlled_actions(sample_medagent_task: MedAgentBenchTask):
    """Verify whitelisted MedAgentBench clinical tools execute without leaking answer."""
    with MedAgentBenchEnvironment(sample_medagent_task) as env:
        # 1. get_patient_vitals
        v_obs = env.execute_action("get_patient_vitals")
        assert v_obs.success is True
        assert v_obs.data["vitals"]["heart_rate"] == "104 bpm"

        # 2. get_lab_results
        l_obs = env.execute_action("get_lab_results", {"category": "cardiac"})
        assert l_obs.success is True
        assert l_obs.data["results"]["bnp"] == "1250 pg/mL"

        # 3. get_imaging_reports
        img_obs = env.execute_action("get_imaging_reports", {"modality": "cxr"})
        assert img_obs.success is True
        assert "pleural effusions" in img_obs.data["report"]

        # 4. get_clinical_history
        h_obs = env.execute_action("get_clinical_history")
        assert h_obs.success is True
        assert len(h_obs.data["differential_diagnoses"]) == 3

        # 5. order_diagnostic_test & get_ordered_tests
        ord_obs = env.execute_action(
            "order_diagnostic_test",
            {"test_name": "Echocardiogram", "rationale": "Assess ejection fraction and valve motion"},
        )
        assert ord_obs.success is True
        assert ord_obs.data["status"] == "ORDERED"

        orders_obs = env.execute_action("get_ordered_tests")
        assert orders_obs.success is True
        assert orders_obs.data["ordered_tests_count"] == 1
        assert orders_obs.data["orders"][0]["test_name"] == "Echocardiogram"


def test_unauthorized_action_rejected(sample_krama_task: KramaBenchTask):
    """Verify unauthorized or malicious tool calls are strictly blocked."""
    with KramaBenchEnvironment(sample_krama_task) as env:
        obs = env.execute_action("execute_arbitrary_shell", {"command": "whoami"})
        assert obs.success is False
        assert "Unauthorized action" in obs.error


# ============================================================================
# 3. Execution Timeout Enforcement Tests
# ============================================================================

def test_execution_timeout_enforcement(sample_krama_task: KramaBenchTask):
    """Verify action exceeding timeout terminates cleanly with timed_out=True."""
    with KramaBenchEnvironment(sample_krama_task, default_timeout_seconds=0.1) as env:
        # Patch a mock slow dispatch
        original_dispatch = env._dispatch_action
        def slow_dispatch(action, params):
            time.sleep(0.3)
            return original_dispatch(action, params)

        env._dispatch_action = slow_dispatch

        obs = env.execute_action("list_data_sources", timeout_seconds=0.08)
        assert obs.success is False
        assert obs.timed_out is True
        assert "timed out" in obs.error.lower()


# ============================================================================
# 4. Observation Capture & Telemetry Tests
# ============================================================================

def test_observation_history_capture(sample_medagent_task: MedAgentBenchTask):
    """Verify tool observations record timing, agent ID, branch ID, and history."""
    with MedAgentBenchEnvironment(sample_medagent_task, branch_id="main") as env:
        env.execute_action("get_patient_vitals", calling_agent_id="agent_cardiologist")
        env.execute_action("get_lab_results", {"category": "cardiac"}, calling_agent_id="agent_cardiologist")

        history = env.get_observation_history()
        assert len(history) == 2
        assert history[0].action == "get_patient_vitals"
        assert history[0].metadata["agent_id"] == "agent_cardiologist"
        assert history[0].metadata["branch_id"] == "main"
        assert history[0].execution_time_ms >= 0.0
        assert history[1].action == "get_lab_results"


# ============================================================================
# 5. Clean Environment Reset Tests
# ============================================================================

def test_clean_environment_reset(sample_krama_task: KramaBenchTask):
    """Verify reset purges generated files, clears observation history, and updates seed."""
    with KramaBenchEnvironment(sample_krama_task, seed=10) as env:
        env.execute_action("write_intermediate_note", {"title": "draft.txt", "content": "Temporary draft"})
        env.execute_action("list_data_sources")
        assert len(env.get_observation_history()) == 2
        assert (env.workspace_dir / "notes_draft.txt").exists()

        # Reset environment with new seed
        clean_state = env.reset(seed=99)
        assert env.seed == 99
        assert len(env.get_observation_history()) == 0
        assert not (env.workspace_dir / "notes_draft.txt").exists()
        assert clean_state.task_id == sample_krama_task.task_id
        assert clean_state.status == BlackboardStatus.ACTIVE


# ============================================================================
# 6. Student 3 Sandbox Coordination Tests
# ============================================================================

def test_student3_sandbox_coordination_and_branch_isolation(sample_medagent_task: MedAgentBenchTask):
    """
    Verify coordination with Student 3's SandboxSession specifications:
    Tool states and database writes/generated files on a sandbox branch must
    remain strictly isolated and never leak back to the parent or live environments.
    """
    with MedAgentBenchEnvironment(sample_medagent_task, branch_id="main") as live_env:
        # Live environment orders 1 baseline test
        live_env.execute_action(
            "order_diagnostic_test",
            {"test_name": "Standard ECG", "rationale": "Baseline rhythm check"},
            calling_agent_id="agent_primary",
        )
        assert len(live_env.execute_action("get_ordered_tests").data["orders"]) == 1

        # Student 3 deadlocks and spawns a counterfactual branch sandbox:
        sandbox_branch_id = "sandbox_cf_a1b2c3d4"
        with live_env.fork_sandbox_environment(sandbox_branch_id=sandbox_branch_id) as cf_env:
            assert cf_env.branch_id == sandbox_branch_id
            # Sandbox inherits the baseline test
            assert len(cf_env.execute_action("get_ordered_tests").data["orders"]) == 1

            # In sandbox exploration, order counterfactual diagnostic tests:
            cf_env.execute_action(
                "order_diagnostic_test",
                {"test_name": "Coronary Angiography", "rationale": "Investigate acute coronary ischemia hypothesis"},
                calling_agent_id="cf_agent_solver",
            )
            cf_env.execute_action(
                "order_diagnostic_test",
                {"test_name": "High-Resolution Chest CT", "rationale": "Rule out pulmonary fibrosis"},
                calling_agent_id="cf_agent_solver",
            )

            # Sandbox now has 3 ordered tests
            cf_orders = cf_env.execute_action("get_ordered_tests").data["orders"]
            assert len(cf_orders) == 3

        # CRITICAL TEST: Live environment must remain completely UNTOUCHED
        live_orders = live_env.execute_action("get_ordered_tests").data["orders"]
        assert len(live_orders) == 1
        assert live_orders[0]["test_name"] == "Standard ECG"


def test_student3_sandbox_file_isolation(sample_krama_task: KramaBenchTask):
    """Verify files created within a counterfactual sandbox branch do not appear in the live workspace."""
    with KramaBenchEnvironment(sample_krama_task, branch_id="main") as live_env:
        live_env.execute_action("write_intermediate_note", {"title": "live_note.txt", "content": "Live blackboard baseline"})

        with live_env.fork_sandbox_environment(sandbox_branch_id="sandbox_cf_test_fork") as cf_env:
            cf_env.execute_action(
                "write_intermediate_note",
                {"title": "hypothetical_calc.txt", "content": "Branch speculative exploration"},
            )
            cf_notes = cf_env.execute_action("read_intermediate_notes").data["notes"]
            assert "notes_hypothetical_calc.txt" in cf_notes

        # Live environment must NOT contain the hypothetical branch file
        live_notes = live_env.execute_action("read_intermediate_notes").data["notes"]
        assert "notes_live_note.txt" in live_notes
        assert "notes_hypothetical_calc.txt" not in live_notes


# ============================================================================
# 7. Diagnostic Checks: Ingestion Safety & Solution Stripping Tests
# ============================================================================

def test_diagnostic_check_kramabench_preservation_and_stripping(sample_krama_task: KramaBenchTask):
    """
    Run diagnostic checks verifying that source task IDs, answer types, and
    constraints are preserved while reference solutions are stripped from agent-visible inputs.
    """
    raw_dict = {
        "id": "krama_wildfire_101",
        "query": "Correlate fuel moisture and wildfire perimeter expansion in the Quebec 2023 season.",
        "domain": "Wildfire Prevention",
        "data_sources": ["fuel_moisture_grid.csv", "satellite_burn_perimeter.parquet"],
        "subtasks": ["Filter perimeter >= 100ha", "Calculate Pearson correlation"],
        "expected_output": {"correlation": -0.82, "p_val": 0.001},
        "ground_truth": '{"correlation": -0.82}',
    }
    adapter = KramaBenchAdapter()
    parsed = adapter.load_from_dict(raw_dict)
    state = adapter.format_for_blackboard(parsed)

    diagnostic = BenchmarkEvaluator.verify_adapter_integrity(raw_dict, parsed, state)
    assert diagnostic.passed is True
    assert diagnostic.source_task_id_preserved is True
    assert diagnostic.answer_types_preserved is True
    assert diagnostic.constraints_preserved is True
    assert diagnostic.reference_solution_stripped_from_agent_input is True

    # Confirm agent-visible initial_context has no ground truth or solution keys
    assert "ground_truth" not in state.initial_context
    assert "expected_output" not in state.initial_context
    assert "solution" not in state.initial_context


def test_diagnostic_check_medagentbench_preservation_and_stripping(sample_medagent_task: MedAgentBenchTask):
    """
    Run diagnostic checks verifying MedAgentBench task integrity and solution stripping.
    """
    raw_dict = {
        "case_id": "case_cardio_202",
        "problem_statement": "64-year-old male presenting with acute dyspnea and elevated BNP.",
        "ground_truth": "Acute Decompensated Heart Failure (ADHF)",
        "patient_id": "PT_64M_99",
        "differential_diagnoses": ["ADHF", "IPF", "PE"],
        "sol": "Acute Decompensated Heart Failure (ADHF)",
        "context": {"vitals": {"bp": "158/94"}},
    }
    adapter = MedAgentBenchAdapter()
    parsed = adapter.load_from_dict(raw_dict)
    state = adapter.format_for_blackboard(parsed)

    diagnostic = BenchmarkEvaluator.verify_adapter_integrity(raw_dict, parsed, state)
    assert diagnostic.passed is True
    assert diagnostic.source_task_id_preserved is True
    assert diagnostic.reference_solution_stripped_from_agent_input is True

    # Crucial: Evaluator can still access ground_truth on task and state
    assert parsed.ground_truth == "Acute Decompensated Heart Failure (ADHF)"
    assert state.ground_truth == "Acute Decompensated Heart Failure (ADHF)"
    # But agent-visible context has NO solution keys
    assert "ground_truth" not in state.initial_context
    assert "sol" not in state.initial_context
    assert "solution" not in state.initial_context


# ============================================================================
# 8. Benchmark Evaluator Operational Tests
# ============================================================================

def test_evaluator_exact_and_normalized_match(sample_medagent_task: MedAgentBenchTask):
    """Verify evaluator matches exact and normalized clinical diagnoses."""
    evaluator = BenchmarkEvaluator()

    # Exact match
    res_exact = evaluator.evaluate(sample_medagent_task, "Acute Decompensated Heart Failure (ADHF)")
    assert res_exact.exact_match is True
    assert res_exact.normalized_match is True
    assert res_exact.token_f1 == 1.0
    assert res_exact.matched_differential == "Acute Decompensated Heart Failure (ADHF)"

    # Normalized match (lowercase with extra words)
    res_norm = evaluator.evaluate(
        sample_medagent_task,
        "Based on elevated BNP and effusions, the patient has acute decompensated heart failure (adhf).",
    )
    assert res_norm.exact_match is False
    assert res_norm.normalized_match is True
    assert res_norm.token_f1 > 0.40
    assert res_norm.matched_differential == "Acute Decompensated Heart Failure (ADHF)"

    # Incorrect / non-matching prediction
    res_wrong = evaluator.evaluate(sample_medagent_task, "Definitive diagnosis is Bacterial Pneumonia.")
    assert res_wrong.exact_match is False
    assert res_wrong.normalized_match is False
    assert res_wrong.matched_differential is None


def test_token_f1_calculation():
    """Verify token F1 calculation across varied prediction overlap."""
    assert compute_token_f1("Acute Heart Failure", "Acute Heart Failure") == 1.0
    assert compute_token_f1("Heart Failure", "Acute Heart Failure") == round(4 / 5, 4)
    assert compute_token_f1("Pulmonary Embolism", "Acute Heart Failure") == 0.0
