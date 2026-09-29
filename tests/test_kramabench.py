"""
tests/test_kramabench.py
Comprehensive unit tests for the KramaBench dataset ingestion adapter.
Verifies parsing of multi-stage data lake discovery queries, domain inference across
the 6 canonical domains, Blackboard state contracts, and TrialConfig matrix generation.
"""

from __future__ import annotations

import json
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
from ingest.kramabench import (
    DEFAULT_DISCOVERY_ROLES,
    KRAMABENCH_DOMAINS,
    KramaBenchAdapter,
    KramaBenchTask,
    infer_kramabench_domain,
    normalize_domain_name,
)
from storage.redis_store import InMemoryRedisClient, RedisBlackboardStore


@pytest.fixture
def adapter() -> KramaBenchAdapter:
    return KramaBenchAdapter()


@pytest.fixture
def sample_krama_task() -> dict:
    return {
        "id": "krama_wildfire_04",
        "query": (
            "Analyze correlation between fuel moisture sensor readings and wildfire perimeter "
            "expansion rates across the 2023 Quebec fire season data lake."
        ),
        "domain": "Wildfire Prevention",
        "data_sources": [
            "fuel_moisture_sensor_grid.csv",
            "satellite_burn_perimeter.parquet",
            "weather_stations_hourly.json",
        ],
        "subtasks": [
            {
                "id": "subtask-1",
                "query": "Align timestamps between sensor grid and satellite perimeter passes.",
                "data_sources": ["fuel_moisture_sensor_grid.csv", "satellite_burn_perimeter.parquet"],
            },
            {
                "id": "subtask-2",
                "query": "Compute Pearson correlation between fuel moisture index and area growth delta.",
                "data_sources": ["satellite_burn_perimeter.parquet", "weather_stations_hourly.json"],
            },
        ],
        "expected_output": {"correlation": -0.78, "p_value": 0.001, "significant": True},
    }


# ============================================================================
# 1. Parsing & Field Normalization Tests
# ============================================================================

def test_parse_standard_kramabench_task(adapter: KramaBenchAdapter, sample_krama_task: dict):
    """Verify standard KramaBench workload item parsing and normalization."""
    task = adapter.load_from_dict(sample_krama_task)

    assert isinstance(task, KramaBenchTask)
    assert task.task_id == "krama_wildfire_04"
    assert task.benchmark_type == BenchmarkType.KRAMABENCH
    assert "Analyze correlation" in task.query
    assert task.problem_statement == task.query
    assert task.domain == "wildfire_prevention"
    assert len(task.data_sources) == 3
    assert "fuel_moisture_sensor_grid.csv" in task.data_sources
    assert len(task.subtasks) == 2
    assert task.expected_output["correlation"] == -0.78
    assert '"correlation": -0.78' in task.ground_truth
    assert task.recommended_roles == DEFAULT_DISCOVERY_ROLES


def test_parse_synonym_fields(adapter: KramaBenchAdapter):
    """Verify adapter handles alternative field naming conventions."""
    raw = {
        "workload_id": "krama_astro_99",
        "instruction": "Identify exoplanet candidate transits in Kepler sector 14 light curves.",
        "files": ["kepler_lightcurve_sec14.fits", "stellar_parameters.csv"],
        "stages": ["Extract raw flux", "Apply Box Least Squares filter"],
        "ground_truth": "Candidate identified at period 4.2 days",
    }
    task = adapter.load_from_dict(raw)

    assert task.task_id == "krama_astro_99"
    assert task.problem_statement == raw["instruction"]
    assert task.query == raw["instruction"]
    assert task.data_sources == raw["files"]
    assert task.subtasks == raw["stages"]
    assert task.ground_truth == "Candidate identified at period 4.2 days"
    assert task.domain == "astronomy"


def test_parse_string_subtask(adapter: KramaBenchAdapter):
    """Verify single string subtask is correctly wrapped into list."""
    raw = {
        "id": "krama_simple_01",
        "query": "Count distinct artifact classifications.",
        "subtasks": "Run SQL aggregation query",
        "domain": "archaeology",
    }
    task = adapter.load_from_dict(raw)
    assert task.subtasks == ["Run SQL aggregation query"]


# ============================================================================
# 2. Domain Normalization & Automatic Inference Tests
# ============================================================================

def test_domain_normalization():
    """Verify domains are mapped into canonical lowercase snake_case standard."""
    assert normalize_domain_name("Wildfire Prevention") == "wildfire_prevention"
    assert normalize_domain_name("ENVIRONMENTAL_SCIENCE") == "environmental_science"
    assert normalize_domain_name("Legal Discovery") == "legal_discovery"
    assert normalize_domain_name("Astronomy") == "astronomy"
    assert normalize_domain_name("Biomedical Research") == "biomedical"
    assert normalize_domain_name("Archaeology") == "archaeology"
    assert normalize_domain_name("") == "general_discovery"
    assert normalize_domain_name(None) == "general_discovery"


def test_infer_canonical_domains():
    """Verify domain inference across all 6 canonical KramaBench domains."""
    # 1. Astronomy
    assert infer_kramabench_domain("Search Kepler light curve files for stellar transit signals.") == "astronomy"

    # 2. Archaeology
    assert infer_kramabench_domain("Analyze excavation strata sediment and pottery carbon dating.") == "archaeology"

    # 3. Biomedical
    assert infer_kramabench_domain("Investigate protein phosphorylation in cellular assay pathways.") == "biomedical"

    # 4. Environmental Science
    assert infer_kramabench_domain("Correlate sensor network CO2 emissions with urban AQI.") == "environmental_science"

    # 5. Legal Discovery
    assert infer_kramabench_domain("Find deposition transcripts concerning patent claim infringement.") == "legal_discovery"

    # 6. Wildfire Prevention
    assert infer_kramabench_domain("Track wildfire burn perimeters and dry fuel moisture indices.") == "wildfire_prevention"


def test_infer_domain_from_data_sources():
    """Verify domain is inferred from file names when query is generic."""
    domain = infer_kramabench_domain(
        query_text="Find relationships between dataset A and dataset B.",
        data_sources=["kepler_stellar_catalog.csv", "telescope_photometry.parquet"],
    )
    assert domain == "astronomy"


# ============================================================================
# 3. Validation & Error Handling Tests
# ============================================================================

def test_missing_task_id_raises_ingestion_error(adapter: KramaBenchAdapter):
    """Verify missing task ID raises IngestionError."""
    with pytest.raises(IngestionError, match="missing a valid identifier"):
        adapter.load_from_dict({"query": "Valid query text without ID"})


def test_empty_query_raises_ingestion_error(adapter: KramaBenchAdapter):
    """Verify empty query raises IngestionError."""
    with pytest.raises(IngestionError, match="must contain non-empty query"):
        adapter.load_from_dict({"id": "task-no-query", "query": "   "})


def test_invalid_dict_type_raises_ingestion_error(adapter: KramaBenchAdapter):
    """Verify non-dictionary input raises IngestionError."""
    with pytest.raises(IngestionError, match="Expected dict input"):
        adapter.load_from_dict(12345)  # type: ignore[arg-type]


def test_malformed_json_raises_ingestion_error(adapter: KramaBenchAdapter):
    """Verify invalid JSON string raises IngestionError."""
    with pytest.raises(IngestionError, match="Invalid JSON content"):
        adapter.load_from_json("{broken: json:")


def test_file_not_found_raises_file_not_found(adapter: KramaBenchAdapter):
    """Verify non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        adapter.load_from_file("non_existent_kramabench_workload.json")


# ============================================================================
# 4. Blackboard Contract Formatting Tests
# ============================================================================

def test_format_for_blackboard_creates_valid_state(adapter: KramaBenchAdapter, sample_krama_task: dict):
    """Verify format_for_blackboard returns BlackboardState ready for architecture."""
    state = adapter.format_for_blackboard(sample_krama_task)

    assert isinstance(state, BlackboardState)
    assert state.session_id.startswith("session_krama_krama_wildfire_04_")
    assert state.task_id == "krama_wildfire_04"
    assert state.status == BlackboardStatus.ACTIVE
    assert "Analyze correlation between fuel moisture" in state.problem_statement
    assert state.metadata["benchmark_type"] == "KramaBench"
    assert state.metadata["domain"] == "wildfire_prevention"
    assert "Data Engineer" in state.metadata["recommended_roles"]
    assert "fuel_moisture_sensor_grid.csv" in state.initial_context["data_sources"]
    assert len(state.initial_context["subtasks"]) == 2
    assert state.initial_context["discovered_relations"] == []


def test_custom_session_id_honored(adapter: KramaBenchAdapter, sample_krama_task: dict):
    """Verify custom session ID is honored when formatting BlackboardState."""
    state = adapter.format_for_blackboard(sample_krama_task, session_id="custom_krama_session")
    assert state.session_id == "custom_krama_session"


def test_blackboard_state_initializes_in_memory_store(adapter: KramaBenchAdapter, sample_krama_task: dict):
    """Verify BlackboardState integrates cleanly with InMemoryBlackboard."""
    state = adapter.format_for_blackboard(sample_krama_task)
    board = InMemoryBlackboard(
        session_id=state.session_id,
        task_id=state.task_id,
        problem_statement=state.problem_statement,
        ground_truth=state.ground_truth,
    )
    current_state = board.get_state()
    assert current_state.session_id == state.session_id
    assert current_state.task_id == state.task_id
    assert current_state.problem_statement == state.problem_statement


def test_blackboard_state_initializes_redis_store(adapter: MedAgentBenchAdapter, sample_krama_task: dict):
    """Verify BlackboardState initializes a session in RedisBlackboardStore (in-memory mode)."""
    krama_adapter = KramaBenchAdapter()
    state = krama_adapter.format_for_blackboard(sample_krama_task)
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
    assert snapshot.initial_context["domain"] == "wildfire_prevention"
    assert "fuel_moisture_sensor_grid.csv" in snapshot.initial_context["data_sources"]


# ============================================================================
# 5. TrialConfig Formatting Tests
# ============================================================================

def test_format_for_trial(adapter: KramaBenchAdapter, sample_krama_task: dict):
    """Verify format_for_trial produces a valid TrialConfig contract."""
    task = adapter.load_from_dict(sample_krama_task)
    config = adapter.format_for_trial(
        task=task,
        counterfactual_density=0.33,
        agent_models={"Data Engineer": "qwen2.5:7b", "Statistician": "llama3.1:8b"},
        max_turns=25,
        seed=77,
    )

    assert isinstance(config, TrialConfig)
    assert config.benchmark_name == BenchmarkType.KRAMABENCH
    assert config.task_id == "krama_wildfire_04"
    assert config.counterfactual_density == 0.33
    assert config.agent_models["Data Engineer"] == "qwen2.5:7b"
    assert config.max_turns == 25
    assert config.seed == 77


def test_format_for_trial_from_dict(adapter: KramaBenchAdapter, sample_krama_task: dict):
    """Verify format_for_trial works directly on a raw dictionary."""
    config = adapter.format_for_trial(sample_krama_task, counterfactual_density=1.0)
    assert config.task_id == "krama_wildfire_04"
    assert config.counterfactual_density == 1.0


# ============================================================================
# 6. Domain Filtering & Batch Utilities Tests
# ============================================================================

def test_filter_by_domain_and_get_domains(adapter: KramaBenchAdapter, sample_krama_task: dict):
    """Verify domain filtering and domain extraction utilities."""
    t1 = adapter.load_from_dict(sample_krama_task)  # wildfire_prevention

    t2_dict = dict(sample_krama_task)
    t2_dict["id"] = "krama_astro_01"
    t2_dict["query"] = "Search star catalog for luminosity variations."
    t2_dict["domain"] = "Astronomy"
    t2 = adapter.load_from_dict(t2_dict)

    tasks = [t1, t2]

    # Test get_domains
    domains = adapter.get_domains(tasks)
    assert domains == ["astronomy", "wildfire_prevention"]

    # Test filter_by_domain
    wildfire_tasks = adapter.filter_by_domain(tasks, "wildfire_prevention")
    assert len(wildfire_tasks) == 1
    assert wildfire_tasks[0].task_id == "krama_wildfire_04"

    astro_tasks = adapter.filter_by_domain(tasks, "astronomy")
    assert len(astro_tasks) == 1
    assert astro_tasks[0].task_id == "krama_astro_01"

    legal_tasks = adapter.filter_by_domain(tasks, "legal_discovery")
    assert len(legal_tasks) == 0


def test_generate_trial_configs_matrix(adapter: KramaBenchAdapter, sample_krama_task: dict):
    """Verify batch trial config generation across ablation densities."""
    task = adapter.load_from_dict(sample_krama_task)
    configs = adapter.generate_trial_configs([task])

    # 4 standard densities: 0.0, 0.33, 0.66, 1.0
    assert len(configs) == 4
    densities = [c.counterfactual_density for c in configs]
    assert densities == [0.0, 0.33, 0.66, 1.0]
    for c in configs:
        assert c.benchmark_name == BenchmarkType.KRAMABENCH
        assert c.task_id == "krama_wildfire_04"


# ============================================================================
# 7. File & Dataset Ingestion Tests
# ============================================================================

def test_load_from_json_file(adapter: KramaBenchAdapter, tmp_path: Path, sample_krama_task: dict):
    """Verify loading from JSON file containing array of tasks."""
    file_path = tmp_path / "krama_tasks.json"
    file_path.write_text(json.dumps([sample_krama_task]), encoding="utf-8")

    loaded = adapter.load_from_file(file_path)
    assert len(loaded) == 1
    assert loaded[0].task_id == "krama_wildfire_04"


def test_load_from_json_container(adapter: KramaBenchAdapter, tmp_path: Path, sample_krama_task: dict):
    """Verify loading from JSON file with {'tasks': [...]} container."""
    file_path = tmp_path / "wrapped_krama.json"
    file_path.write_text(json.dumps({"tasks": [sample_krama_task]}), encoding="utf-8")

    loaded = adapter.load_from_file(file_path)
    assert len(loaded) == 1
    assert loaded[0].task_id == "krama_wildfire_04"


def test_load_from_jsonl_file(adapter: KramaBenchAdapter, tmp_path: Path, sample_krama_task: dict):
    """Verify loading from JSONL file with one task per line."""
    file_path = tmp_path / "krama_workload.jsonl"
    task2 = dict(sample_krama_task)
    task2["id"] = "krama_wildfire_05"

    content = f"{json.dumps(sample_krama_task)}\n{json.dumps(task2)}\n"
    file_path.write_text(content, encoding="utf-8")

    loaded = adapter.load_from_file(file_path)
    assert len(loaded) == 2
    assert loaded[0].task_id == "krama_wildfire_04"
    assert loaded[1].task_id == "krama_wildfire_05"


def test_load_dataset_directory(adapter: KramaBenchAdapter, tmp_path: Path, sample_krama_task: dict):
    """Verify loading multiple JSON workload files from a directory."""
    f1 = tmp_path / "workload_1.json"
    f2 = tmp_path / "workload_2.json"

    task2 = dict(sample_krama_task)
    task2["id"] = "krama_dir_02"

    f1.write_text(json.dumps([sample_krama_task]), encoding="utf-8")
    f2.write_text(json.dumps([task2]), encoding="utf-8")

    all_tasks = adapter.load_dataset(tmp_path)
    assert len(all_tasks) == 2
    ids = {t.task_id for t in all_tasks}
    assert ids == {"krama_wildfire_04", "krama_dir_02"}


# ============================================================================
# 8. Serialization Roundtrip Tests
# ============================================================================

def test_kramabench_task_serialization_roundtrip(adapter: KramaBenchAdapter, sample_krama_task: dict):
    """Verify Pydantic model serialization and roundtrip integrity."""
    task = adapter.load_from_dict(sample_krama_task)
    json_str = task.model_dump_json()

    reconstructed = KramaBenchTask.model_validate_json(json_str)
    assert reconstructed.task_id == task.task_id
    assert reconstructed.query == task.query
    assert reconstructed.problem_statement == task.problem_statement
    assert reconstructed.domain == task.domain
    assert reconstructed.data_sources == task.data_sources
    assert reconstructed.ground_truth == task.ground_truth
    assert reconstructed.expected_output == task.expected_output
