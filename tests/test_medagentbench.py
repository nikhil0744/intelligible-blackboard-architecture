"""
tests/test_medagentbench.py
Comprehensive unit tests for the MedAgentBench dataset ingestion adapter.
Verifies parsing of clinical vignettes, EHR records, specialist role inference,
Blackboard contract formatting, and Section 2.5 ablation resampling.
"""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
import pytest

from blackboard.store import InMemoryBlackboard
from contracts.schemas import (
    BenchmarkType,
    BlackboardSnapshot,
    BlackboardState,
    BlackboardStatus,
    TrialConfig,
)
from ingest.base import IngestionError
from ingest.medagentbench import (
    DEFAULT_CLINICAL_SPECIALTIES,
    MedAgentBenchAdapter,
    MedAgentBenchTask,
    infer_clinical_specialties,
)
from storage.redis_store import InMemoryRedisClient, RedisBlackboardStore


@pytest.fixture
def adapter() -> MedAgentBenchAdapter:
    return MedAgentBenchAdapter()


@pytest.fixture
def sample_vignette() -> dict:
    return {
        "id": "med_task_cardio_pulmo_01",
        "instruction": (
            "62-year-old female presents with acute dyspnea, orthopnea, and bilateral lower "
            "extremity edema. CXR reveals cardiomegaly and pulmonary congestion. BNP is 1200 pg/mL. "
            "Determine primary diagnosis and multidisciplinary stabilization plan."
        ),
        "context": {
            "vitals": {"bp": "158/94", "hr": 98, "rr": 24, "spo2": "89% on RA"},
            "labs": {"bnp": 1200, "troponin": 0.02, "creatinine": 1.4},
            "imaging": "Bilateral bat-wing alveolar opacities consistent with cardiogenic pulmonary edema",
        },
        "sol": ["Acute Decompensated Heart Failure (ADHF)"],
        "eval_MRN": "MRN-90210",
        "task_type": "diagnostic_dilemma",
    }


# ============================================================================
# 1. Parsing & Field Normalization Tests
# ============================================================================

def test_parse_standard_medagentbench_task(adapter: MedAgentBenchAdapter, sample_vignette: dict):
    """Verify standard MedAgentBench record parsing and field normalization."""
    task = adapter.load_from_dict(sample_vignette)

    assert isinstance(task, MedAgentBenchTask)
    assert task.task_id == "med_task_cardio_pulmo_01"
    assert task.benchmark_type == BenchmarkType.MEDAGENTBENCH
    assert "acute dyspnea" in task.problem_statement
    assert task.ground_truth == "Acute Decompensated Heart Failure (ADHF)"
    assert task.patient_id == "MRN-90210"
    assert task.eval_mrn == "MRN-90210"
    assert task.sol == ["Acute Decompensated Heart Failure (ADHF)"]
    assert task.patient_profile["labs"]["bnp"] == 1200
    assert task.task_type == "diagnostic_dilemma"


def test_parse_synonym_fields(adapter: MedAgentBenchAdapter):
    """Verify adapter handles alternative naming conventions across benchmark versions."""
    raw = {
        "case_id": "case_onc_042",
        "prompt": "50yo male with solitary pulmonary nodule and hemoptysis. Histology pending.",
        "solution": "Non-Small Cell Lung Carcinoma",
        "patient_id": "MRN-554433",
        "differentials": ["NSCLC", "Tuberculosis", "Hamartoma"],
        "specialties": ["Oncologist", "Pulmonologist", "Pathologist"],
    }
    task = adapter.load_from_dict(raw)

    assert task.task_id == "case_onc_042"
    assert task.problem_statement == raw["prompt"]
    assert task.ground_truth == "Non-Small Cell Lung Carcinoma"
    assert task.patient_id == "MRN-554433"
    assert task.differential_diagnoses == ["NSCLC", "Tuberculosis", "Hamartoma"]
    assert task.clinical_specialties == ["Oncologist", "Pulmonologist", "Pathologist"]


def test_parse_string_context(adapter: MedAgentBenchAdapter):
    """Verify string narrative context is stored in patient_profile under clinical_notes."""
    raw = {
        "id": "case_text_context",
        "instruction": "Evaluate symptoms.",
        "context": "Patient has 3-week history of nocturnal dry cough and low-grade fevers.",
    }
    task = adapter.load_from_dict(raw)
    assert task.patient_profile["clinical_notes"] == raw["context"]


def test_parse_clinical_fixture_sample_deadlocks(adapter: MedAgentBenchAdapter):
    """Verify adapter can ingest fixtures/sample_deadlocks.json without schema friction."""
    fixture_path = files("fixtures").joinpath("sample_deadlocks.json")
    with fixture_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    task = adapter.load_from_dict(data)
    assert task.task_id == "med_task_fibrosis_vs_chf"
    assert task.ground_truth == "Idiopathic Pulmonary Fibrosis (IPF)"
    assert "honeycombing" in task.problem_statement
    assert "Cardiologist" in task.clinical_specialties
    assert "Pulmonologist" in task.clinical_specialties


# ============================================================================
# 2. Specialist Role Inference Tests
# ============================================================================

def test_infer_clinical_specialties_cardio_pulmo():
    """Verify keyword inference maps symptoms to appropriate specialists."""
    text = "Patient exhibits worsening dyspnea, end-inspiratory crackles, and elevated BNP."
    roles = infer_clinical_specialties(text)
    assert "Cardiologist" in roles
    assert "Pulmonologist" in roles


def test_infer_clinical_specialties_oncology_pathology():
    """Verify tumor/biopsy keywords trigger Oncologist and Pathologist personas."""
    text = "Patient has mediastinal mass discovered on CT; needle biopsy reveals malignant cells."
    roles = infer_clinical_specialties(text)
    assert "Oncologist" in roles
    assert "Pathologist" in roles
    assert "Radiologist" in roles


def test_infer_clinical_specialties_fallback_default():
    """Verify fallback to standard clinical panel when keywords are sparse."""
    text = "General evaluation of non-specific weakness and fatigue."
    roles = infer_clinical_specialties(text)
    for expected in DEFAULT_CLINICAL_SPECIALTIES:
        assert expected in roles


# ============================================================================
# 3. Validation & Error Handling Tests
# ============================================================================

def test_missing_task_id_raises_ingestion_error(adapter: MedAgentBenchAdapter):
    """Verify missing task ID raises IngestionError."""
    with pytest.raises(IngestionError, match="missing a valid identifier"):
        adapter.load_from_dict({"instruction": "Missing ID scenario"})


def test_empty_problem_statement_raises_ingestion_error(adapter: MedAgentBenchAdapter):
    """Verify empty clinical problem statement raises IngestionError."""
    with pytest.raises(IngestionError, match="must contain non-empty clinical problem"):
        adapter.load_from_dict({"id": "task-empty", "instruction": "   "})


def test_invalid_dict_type_raises_ingestion_error(adapter: MedAgentBenchAdapter):
    """Verify non-dictionary input raises IngestionError."""
    with pytest.raises(IngestionError, match="Expected dict input"):
        adapter.load_from_dict("not-a-dict")  # type: ignore[arg-type]


def test_malformed_json_raises_ingestion_error(adapter: MedAgentBenchAdapter):
    """Verify invalid JSON string raises IngestionError."""
    with pytest.raises(IngestionError, match="Invalid JSON content"):
        adapter.load_from_json("{broken json")


def test_file_not_found_raises_file_not_found(adapter: MedAgentBenchAdapter):
    """Verify non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        adapter.load_from_file("non_existent_medagent_file.json")


# ============================================================================
# 4. Blackboard Contract Formatting Tests
# ============================================================================

def test_format_for_blackboard_creates_valid_state(adapter: MedAgentBenchAdapter, sample_vignette: dict):
    """Verify format_for_blackboard returns BlackboardState ready for architecture."""
    state = adapter.format_for_blackboard(sample_vignette)

    assert isinstance(state, BlackboardState)
    assert state.session_id.startswith("session_medagent_med_task_cardio_pulmo_01_")
    assert state.task_id == "med_task_cardio_pulmo_01"
    assert state.status == BlackboardStatus.ACTIVE
    assert state.ground_truth == "Acute Decompensated Heart Failure (ADHF)"
    assert state.metadata["benchmark_type"] == "MedAgentBench"
    assert "Cardiologist" in state.metadata["clinical_specialties"]
    assert state.initial_context["patient_id"] == "MRN-90210"
    assert state.initial_context["patient_profile"]["labs"]["bnp"] == 1200


def test_custom_session_id_honored(adapter: MedAgentBenchAdapter, sample_vignette: dict):
    """Verify custom session ID is honored when formatting BlackboardState."""
    state = adapter.format_for_blackboard(sample_vignette, session_id="custom_sess_123")
    assert state.session_id == "custom_sess_123"


def test_blackboard_state_initializes_in_memory_store(adapter: MedAgentBenchAdapter, sample_vignette: dict):
    """Verify BlackboardState integrates cleanly with InMemoryBlackboard."""
    state = adapter.format_for_blackboard(sample_vignette)
    board = InMemoryBlackboard(
        session_id=state.session_id,
        task_id=state.task_id,
        problem_statement=state.problem_statement,
        ground_truth=state.ground_truth,
    )
    current_state = board.get_state()
    assert current_state.session_id == state.session_id
    assert current_state.task_id == state.task_id
    assert current_state.ground_truth == state.ground_truth


def test_blackboard_state_initializes_redis_store(adapter: MedAgentBenchAdapter, sample_vignette: dict):
    """Verify BlackboardState initializes a session in RedisBlackboardStore (in-memory mode)."""
    state = adapter.format_for_blackboard(sample_vignette)
    mock_redis = InMemoryRedisClient()
    store = RedisBlackboardStore(redis_client=mock_redis)

    snapshot = store.create_session(
        session_id=state.session_id,
        task_id=state.task_id,
        task_description=state.task_description,
        ground_truth=state.ground_truth,
        initial_context=state.initial_context,
        metadata=state.metadata,
    )

    assert snapshot.session_id == state.session_id
    assert snapshot.task_id == state.task_id
    assert snapshot.ground_truth == state.ground_truth
    assert snapshot.initial_context["patient_id"] == "MRN-90210"


# ============================================================================
# 5. TrialConfig Formatting Tests
# ============================================================================

def test_format_for_trial(adapter: MedAgentBenchAdapter, sample_vignette: dict):
    """Verify format_for_trial produces a valid TrialConfig contract."""
    task = adapter.load_from_dict(sample_vignette)
    config = adapter.format_for_trial(
        task=task,
        counterfactual_density=0.66,
        agent_models={"Cardiologist": "qwen2.5:7b", "Pulmonologist": "llama3.1:8b"},
        max_turns=20,
        seed=101,
    )

    assert isinstance(config, TrialConfig)
    assert config.benchmark_name == BenchmarkType.MEDAGENTBENCH
    assert config.task_id == "med_task_cardio_pulmo_01"
    assert config.counterfactual_density == 0.66
    assert config.agent_models["Cardiologist"] == "qwen2.5:7b"
    assert config.max_turns == 20
    assert config.seed == 101


def test_format_for_trial_from_dict(adapter: MedAgentBenchAdapter, sample_vignette: dict):
    """Verify format_for_trial can parse directly from raw dictionary."""
    config = adapter.format_for_trial(sample_vignette, counterfactual_density=0.33)
    assert config.task_id == "med_task_cardio_pulmo_01"
    assert config.counterfactual_density == 0.33


# ============================================================================
# 6. Section 2.5 Resampling & Ablation Expansion Tests
# ============================================================================

def test_resample_for_ablation_expansion(adapter: MedAgentBenchAdapter, sample_vignette: dict):
    """
    Verify Section 2.5 resampling expands smaller task collections to required trial targets
    while creating unique IDs and permuting role seeds.
    """
    task1 = adapter.load_from_dict(sample_vignette)
    task2_dict = dict(sample_vignette)
    task2_dict["id"] = "med_task_pulmo_02"
    task2 = adapter.load_from_dict(task2_dict)

    tasks = [task1, task2]
    target_count = 10
    resampled = adapter.resample_for_ablation(
        tasks=tasks,
        target_count=target_count,
        seed=42,
        role_permutations=True,
    )

    assert len(resampled) == target_count
    # All task IDs must be distinct
    task_ids = [t.task_id for t in resampled]
    assert len(set(task_ids)) == target_count

    # Verify metadata tracking
    for idx, t in enumerate(resampled):
        assert t.metadata["resample_index"] == idx
        assert "role_seed" in t.metadata
        assert t.metadata["source_task_id"] in ["med_task_cardio_pulmo_01", "med_task_pulmo_02"]


def test_resample_reproducibility(adapter: MedAgentBenchAdapter, sample_vignette: dict):
    """Verify resampling is strictly reproducible with the same seed."""
    task = adapter.load_from_dict(sample_vignette)
    resampled_a = adapter.resample_for_ablation([task], target_count=5, seed=99)
    resampled_b = adapter.resample_for_ablation([task], target_count=5, seed=99)

    assert [t.task_id for t in resampled_a] == [t.task_id for t in resampled_b]
    assert [t.clinical_specialties for t in resampled_a] == [t.clinical_specialties for t in resampled_b]


def test_resample_invalid_inputs(adapter: MedAgentBenchAdapter):
    """Verify error checking on empty tasks or invalid target count."""
    with pytest.raises(IngestionError, match="Cannot resample an empty"):
        adapter.resample_for_ablation([], target_count=10)

    task = MedAgentBenchTask(
        task_id="t1",
        problem_statement="Test statement",
    )
    with pytest.raises(IngestionError, match="must be positive"):
        adapter.resample_for_ablation([task], target_count=0)


def test_generate_trial_configs_matrix(adapter: MedAgentBenchAdapter, sample_vignette: dict):
    """Verify trial config matrix generation across standard ablation densities."""
    task = adapter.load_from_dict(sample_vignette)
    configs = adapter.generate_trial_configs([task])

    # 4 standard densities: 0.0, 0.33, 0.66, 1.0
    assert len(configs) == 4
    densities = [c.counterfactual_density for c in configs]
    assert densities == [0.0, 0.33, 0.66, 1.0]


# ============================================================================
# 7. File & Dataset Ingestion Tests
# ============================================================================

def test_load_from_json_file(adapter: MedAgentBenchAdapter, tmp_path: Path, sample_vignette: dict):
    """Verify loading from JSON file containing array of tasks."""
    file_path = tmp_path / "med_tasks.json"
    file_path.write_text(json.dumps([sample_vignette]), encoding="utf-8")

    loaded = adapter.load_from_file(file_path)
    assert len(loaded) == 1
    assert loaded[0].task_id == "med_task_cardio_pulmo_01"


def test_load_from_json_container(adapter: MedAgentBenchAdapter, tmp_path: Path, sample_vignette: dict):
    """Verify loading from JSON file with {'tasks': [...]} container."""
    file_path = tmp_path / "wrapped_tasks.json"
    file_path.write_text(json.dumps({"tasks": [sample_vignette]}), encoding="utf-8")

    loaded = adapter.load_from_file(file_path)
    assert len(loaded) == 1
    assert loaded[0].task_id == "med_task_cardio_pulmo_01"


def test_load_from_jsonl_file(adapter: MedAgentBenchAdapter, tmp_path: Path, sample_vignette: dict):
    """Verify loading from JSONL file with one task per line."""
    file_path = tmp_path / "med_tasks.jsonl"
    vignette2 = dict(sample_vignette)
    vignette2["id"] = "med_task_02"

    content = f"{json.dumps(sample_vignette)}\n{json.dumps(vignette2)}\n"
    file_path.write_text(content, encoding="utf-8")

    loaded = adapter.load_from_file(file_path)
    assert len(loaded) == 2
    assert loaded[0].task_id == "med_task_cardio_pulmo_01"
    assert loaded[1].task_id == "med_task_02"


def test_load_dataset_directory(adapter: MedAgentBenchAdapter, tmp_path: Path, sample_vignette: dict):
    """Verify loading entire dataset directory with multiple JSON files."""
    f1 = tmp_path / "batch_1.json"
    f2 = tmp_path / "batch_2.json"

    vignette2 = dict(sample_vignette)
    vignette2["id"] = "med_batch2_01"

    f1.write_text(json.dumps([sample_vignette]), encoding="utf-8")
    f2.write_text(json.dumps([vignette2]), encoding="utf-8")

    all_tasks = adapter.load_dataset(tmp_path)
    assert len(all_tasks) == 2
    ids = {t.task_id for t in all_tasks}
    assert ids == {"med_task_cardio_pulmo_01", "med_batch2_01"}


# ============================================================================
# 8. Serialization Roundtrip Tests
# ============================================================================

def test_medagentbench_task_serialization_roundtrip(adapter: MedAgentBenchAdapter, sample_vignette: dict):
    """Verify Pydantic model serialization and roundtrip integrity."""
    task = adapter.load_from_dict(sample_vignette)
    json_str = task.model_dump_json()

    reconstructed = MedAgentBenchTask.model_validate_json(json_str)
    assert reconstructed.task_id == task.task_id
    assert reconstructed.problem_statement == task.problem_statement
    assert reconstructed.ground_truth == task.ground_truth
    assert reconstructed.clinical_specialties == task.clinical_specialties
    assert reconstructed.patient_profile == task.patient_profile
