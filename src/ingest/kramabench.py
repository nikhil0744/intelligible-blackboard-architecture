"""
ingest/kramabench.py
Dataset ingestion adapter for KramaBench multi-agent information discovery and distributed search.
Parses multi-stage data lake discovery tasks across 6 canonical domains into Intelligible
Blackboard Architecture Pydantic contracts.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set, Union
from uuid import uuid4

from pydantic import ConfigDict, Field, model_validator

from contracts.schemas import (
    BenchmarkTask,
    BenchmarkType,
    BlackboardSnapshot,
    BlackboardState,
    BlackboardStatus,
    TrialConfig,
)
from ingest.base import BaseBenchmarkAdapter, IngestionError

# Canonical KramaBench domains (ICLR 2026 / MIT DSG)
KRAMABENCH_DOMAINS: List[str] = [
    "archaeology",
    "astronomy",
    "biomedical",
    "environmental_science",
    "legal_discovery",
    "wildfire_prevention",
]

DEFAULT_DISCOVERY_ROLES: List[str] = [
    "Data Engineer",
    "Domain Researcher",
    "Statistician",
    "Validation Critic",
]

DOMAIN_KEYWORD_MAP: Dict[str, List[str]] = {
    "archaeology": [
        "archaeolog", "excavation", "artifact", "carbon dating", "strata",
        "pottery", "sediment", "burial", "fossil", "tomb",
    ],
    "astronomy": [
        "astronom", "telescope", "light curve", "transit", "star",
        "planet", "galaxy", "stellar", "exoplanet", "kepler", "spectrum",
    ],
    "biomedical": [
        "biomed", "gene", "protein", "dna", "rna", "mutation", "expression",
        "assay", "clinical trial", "cellular", "pathway",
    ],
    "environmental_science": [
        "environment", "emission", "climate", "temperature", "co2",
        "ozone", "pollution", "air quality", "aqi", "precipitation",
        "sensor network", "hydrology",
    ],
    "legal_discovery": [
        "legal", "patent", "deposition", "litigation", "court", "plaintiff",
        "defendant", "statute", "claim", "infringement", "discovery",
    ],
    "wildfire_prevention": [
        "wildfire", "fire", "burn", "fuel moisture", "smoke", "perimeter",
        "ndvi", "ignition", "vegetation", "wind speed",
    ],
}


def normalize_domain_name(domain: Optional[str]) -> str:
    """
    Normalize domain name into lowercase snake_case standard matching canonical domains.
    """
    if not domain or not domain.strip():
        return "general_discovery"

    clean = re.sub(r"[\s\-_]+", "_", domain.strip().lower())
    # Match against known canonicals
    for canonical in KRAMABENCH_DOMAINS:
        if clean == canonical or clean.replace("_", "") == canonical.replace("_", ""):
            return canonical
        if clean.startswith(canonical) or canonical in clean:
            return canonical

    return clean


def infer_kramabench_domain(query_text: str, data_sources: Optional[List[str]] = None) -> str:
    """
    Infer canonical KramaBench domain from query text or data source filenames.
    """
    corpus = (query_text or "").lower()
    if data_sources:
        corpus += " " + " ".join(str(s).lower() for s in data_sources)

    best_domain = "general_discovery"
    max_hits = 0

    for domain, keywords in DOMAIN_KEYWORD_MAP.items():
        hits = sum(1 for kw in keywords if kw in corpus)
        if hits > max_hits:
            max_hits = hits
            best_domain = domain

    return best_domain


class KramaBenchTask(BenchmarkTask):
    """
    Standardized Pydantic contract for a KramaBench information discovery task.
    Represents an unstructured data lake search problem requiring collaborative
    multi-agent exploration without query duplication or write collisions.
    """
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    task_id: str = Field(..., description="Unique KramaBench task identifier.")
    benchmark_type: BenchmarkType = Field(
        default=BenchmarkType.KRAMABENCH,
        description="Benchmark identifier."
    )
    problem_statement: str = Field(
        ...,
        description="Discovery query, hypothesis to verify, or multi-stage research goal."
    )
    query: str = Field(
        ...,
        description="Core search or data exploration query."
    )
    domain: str = Field(
        default="general_discovery",
        description="Domain of the task (e.g. astronomy, wildfire_prevention)."
    )
    data_sources: List[str] = Field(
        default_factory=list,
        description="Data lake files, tables, or document collections involved in the search."
    )
    subtasks: List[Union[str, Dict[str, Any]]] = Field(
        default_factory=list,
        description="Ordered sequence of sub-queries or pipeline discovery stages."
    )
    ground_truth: Optional[str] = Field(
        default=None,
        description="Reference discovery finding, numeric insight, or pipeline verification outcome."
    )
    expected_output: Optional[Any] = Field(
        default=None,
        description="Raw structured or numeric target output."
    )
    recommended_roles: List[str] = Field(
        default_factory=list,
        description="Recommended specialist roles (Data Engineer, Statistician, etc.)."
    )
    initial_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Contextual payload populated into BlackboardState."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Raw dataset metadata, pipeline complexity, or data lake root."
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_kramabench_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        d = dict(data)

        # 1. Normalize task_id
        task_id = (
            d.get("task_id")
            or d.get("id")
            or d.get("workload_id")
            or d.get("main_task_id")
        )
        if not task_id:
            raise ValueError("Task is missing a valid identifier ('task_id', 'id', or 'workload_id').")
        d["task_id"] = str(task_id).strip()

        # 2. Normalize query and problem_statement
        query = (
            d.get("query")
            or d.get("problem_statement")
            or d.get("instruction")
            or d.get("question")
            or d.get("goal")
        )
        if not query or not str(query).strip():
            raise ValueError("KramaBench task must contain non-empty query or problem statement.")
        query_str = str(query).strip()
        d["query"] = query_str
        d["problem_statement"] = query_str

        # 3. Benchmark type
        d["benchmark_type"] = BenchmarkType.KRAMABENCH

        # 4. Data sources
        sources = (
            d.get("data_sources")
            or d.get("files")
            or d.get("tables")
            or d.get("data_lake")
            or []
        )
        if isinstance(sources, list):
            d["data_sources"] = [str(s).strip() for s in sources if str(s).strip()]
        elif isinstance(sources, str) and sources.strip():
            d["data_sources"] = [sources.strip()]
        else:
            d["data_sources"] = []

        # 5. Domain
        raw_domain = d.get("domain")
        if raw_domain and str(raw_domain).strip():
            d["domain"] = normalize_domain_name(str(raw_domain))
        else:
            d["domain"] = infer_kramabench_domain(
                query_text=query_str,
                data_sources=d["data_sources"],
            )

        # 6. Subtasks / Pipeline steps
        subtasks = (
            d.get("subtasks")
            or d.get("pipeline_steps")
            or d.get("stages")
            or d.get("steps")
            or []
        )
        if isinstance(subtasks, list):
            d["subtasks"] = subtasks
        else:
            d["subtasks"] = [subtasks]

        # 7. Ground truth and expected output
        expected = d.get("expected_output") or d.get("ground_truth") or d.get("solution") or d.get("target_insight")
        d["expected_output"] = expected
        if expected is not None:
            if isinstance(expected, (dict, list)):
                d["ground_truth"] = json.dumps(expected)
            else:
                d["ground_truth"] = str(expected).strip()
        else:
            d["ground_truth"] = None

        # 8. Recommended roles
        roles = d.get("recommended_roles") or d.get("roles")
        if roles and isinstance(roles, list):
            d["recommended_roles"] = [str(r).strip() for r in roles if str(r).strip()]
        else:
            d["recommended_roles"] = list(DEFAULT_DISCOVERY_ROLES)

        # 9. Initial Context preparation for BlackboardState
        initial_context = dict(d.get("initial_context") or {})
        initial_context.setdefault("domain", d["domain"])
        initial_context.setdefault("data_sources", d["data_sources"])
        initial_context.setdefault("subtasks", d["subtasks"])
        initial_context.setdefault("discovered_relations", [])
        initial_context.setdefault("query_history", [])
        d["initial_context"] = initial_context

        # 10. Metadata
        meta = dict(d.get("metadata") or {})
        meta.setdefault("benchmark_type", BenchmarkType.KRAMABENCH.value)
        meta.setdefault("domain", d["domain"])
        meta.setdefault("data_source_count", len(d["data_sources"]))
        meta.setdefault("subtask_count", len(d["subtasks"]))
        d["metadata"] = meta

        return d

    def to_blackboard_state(self, session_id: Optional[str] = None) -> BlackboardState:
        """
        Instantiate a live BlackboardState pre-configured for distributed search.
        Initializes data sources and search tracking context.
        """
        sid = session_id or f"session_krama_{self.task_id}_{uuid4().hex[:8]}"
        meta = dict(self.metadata)
        meta["benchmark_type"] = self.benchmark_type.value
        meta["domain"] = self.domain
        meta["recommended_roles"] = list(self.recommended_roles)
        meta["data_sources"] = list(self.data_sources)

        return BlackboardState(
            session_id=sid,
            task_id=self.task_id,
            problem_statement=self.problem_statement,
            ground_truth=self.ground_truth,
            initial_context=dict(self.initial_context),
            status=BlackboardStatus.ACTIVE,
            entries=[],
            active_branches=["main"],
            metadata=meta,
        )

    def to_trial_config(
        self,
        counterfactual_density: float = 0.0,
        agent_models: Optional[Dict[str, str]] = None,
        max_turns: int = 15,
        seed: int = 42,
    ) -> TrialConfig:
        """
        Generate a TrialConfig contract for automated ablation trials.
        """
        models = dict(agent_models or {})
        if not models and self.recommended_roles:
            for role in self.recommended_roles:
                models[role] = "default-discovery-model"

        return TrialConfig(
            benchmark_name=BenchmarkType.KRAMABENCH,
            task_id=self.task_id,
            counterfactual_density=counterfactual_density,
            agent_models=models,
            max_turns=max_turns,
            seed=seed,
        )


class KramaBenchAdapter(BaseBenchmarkAdapter[KramaBenchTask]):
    """
    Ingestion adapter for KramaBench multi-agent data lake exploration benchmark.
    Parses complex data lake queries, maps pipeline subtasks, infers domains,
    and formats tasks into Blackboard and Scheduler contracts.
    """

    task_cls = KramaBenchTask

    def parse_task(self, data: Dict[str, Any]) -> KramaBenchTask:
        """
        Parse and validate a dictionary into a typed KramaBenchTask contract.
        """
        return KramaBenchTask.model_validate(data)

    def filter_by_domain(self, tasks: List[KramaBenchTask], domain: str) -> List[KramaBenchTask]:
        """
        Filter tasks by normalized canonical domain (e.g. 'astronomy', 'wildfire_prevention').
        """
        normalized = normalize_domain_name(domain)
        return [t for t in tasks if t.domain == normalized]

    def get_domains(self, tasks: List[KramaBenchTask]) -> List[str]:
        """
        Get unique list of domains present across ingested tasks.
        """
        return sorted(list({t.domain for t in tasks}))

    def infer_domain(self, query_or_text: str, data_sources: Optional[List[str]] = None) -> str:
        """
        Extract domain from text or file references.
        """
        return infer_kramabench_domain(query_text=query_or_text, data_sources=data_sources)

    def generate_trial_configs(
        self,
        tasks: List[KramaBenchTask],
        counterfactual_densities: Optional[List[float]] = None,
        agent_models: Optional[Dict[str, str]] = None,
        max_turns: int = 15,
        seed: int = 42,
    ) -> List[TrialConfig]:
        """
        Generate trial configuration matrix across all tasks and ablation densities
        (0%, 33%, 66%, 100% counterfactual-capable agents).
        """
        densities = counterfactual_densities if counterfactual_densities is not None else [0.0, 0.33, 0.66, 1.0]
        configs: List[TrialConfig] = []

        for density in densities:
            for task in tasks:
                configs.append(
                    task.to_trial_config(
                        counterfactual_density=density,
                        agent_models=agent_models,
                        max_turns=max_turns,
                        seed=seed,
                    )
                )

        return configs
