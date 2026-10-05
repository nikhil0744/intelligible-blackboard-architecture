"""
ingest/medagentbench.py
Dataset ingestion adapter for MedAgentBench clinical reasoning and diagnostic dilemmas.
Converts clinical vignettes, FHIR/EHR patient contexts, and multidisciplinary consensus tasks
into Intelligible Blackboard Architecture Pydantic contracts.
"""

from __future__ import annotations

import copy
import random
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
from ingest.base import BaseBenchmarkAdapter, IngestionError, strip_reference_solutions

DEFAULT_CLINICAL_SPECIALTIES: List[str] = [
    "Cardiologist",
    "Pulmonologist",
    "Radiologist",
]

SPECIALTY_KEYWORD_MAP: Dict[str, List[str]] = {
    "Cardiologist": [
        "heart", "cardio", "chf", "bnp", "troponin", "ecg", "echo",
        "edema", "hypertension", "murmur", "ejection fraction", "myocardial",
        "arrhythmia", "coronary", "ischemia", "pericard",
    ],
    "Pulmonologist": [
        "lung", "pulmo", "dyspnea", "cough", "crackles", "honeycombing",
        "copd", "fibrosis", "fev1", "wheeze", "respiratory", "ipf",
        "velcro", "reticular", "bronch", "pleural", "hypox",
    ],
    "Radiologist": [
        "radiolog", "ct", "ct scan", "hrct", "x-ray", "cxr", "mri",
        "ultrasound", "scan", "consolidation", "infiltrate", "radiograph",
        "imaging", "opacit", "ground-glass", "lesion", "pet scan",
    ],
    "Oncologist": [
        "oncolog", "tumor", "mass", "malignan", "cancer", "biopsy",
        "stage", "carcinoma", "neoplasm", "chemotherapy", "metast",
        "lymphoma", "melanoma", "sarcoma",
    ],
    "Pathologist": [
        "patholog", "biopsy", "histology", "specimen", "cytology",
        "immunohistochemistry", "microscop",
    ],
    "Nephrologist": [
        "nephro", "kidney", "renal", "creatinine", "egfr", "proteinuria",
        "dialysis", "glomerul",
    ],
    "Neurologist": [
        "neuro", "stroke", "seizure", "headache", "aphasia", "neuropathy",
        "encephal", "mri brain", "cranial",
    ],
    "Infectious Disease Specialist": [
        "infection", "sepsis", "bacterial", "viral", "fever", "antibiotic",
        "culture", "pathogen", "wbc", "procalcitonin",
    ],
}


def infer_clinical_specialties(text: str, context: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Heuristically infer clinical specialist roles required for a multidisciplinary
    consensus panel based on clinical presentation and context keywords.
    """
    corpus = (text or "").lower()
    if context:
        corpus += " " + str(context).lower()

    matched: List[str] = []
    for specialty, keywords in SPECIALTY_KEYWORD_MAP.items():
        found = False
        for kw in keywords:
            if len(kw) <= 3:
                if re.search(r'\b' + re.escape(kw) + r'\b', corpus):
                    found = True
                    break
            elif kw in corpus:
                found = True
                break
        if found:
            matched.append(specialty)

    # Always ensure at least 2 specialists for debate/consensus; if fewer, merge with default
    if len(matched) < 2:
        for s in DEFAULT_CLINICAL_SPECIALTIES:
            if s not in matched:
                matched.append(s)

    return matched


class MedAgentBenchTask(BenchmarkTask):
    """
    Standardized Pydantic contract for a MedAgentBench clinical reasoning task.
    Represents a patient dilemma, tumor board conference, or diagnostic consensus item.
    """
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    task_id: str = Field(..., description="Unique task or case identifier.")
    benchmark_type: BenchmarkType = Field(
        default=BenchmarkType.MEDAGENTBENCH,
        description="Benchmark identifier."
    )
    problem_statement: str = Field(
        ...,
        description="Clinical vignette, patient presentation, or diagnostic question."
    )
    ground_truth: Optional[str] = Field(
        default=None,
        description="Validated reference diagnosis or clinical solution."
    )
    patient_id: Optional[str] = Field(
        default=None,
        description="Patient ID or Medical Record Number (MRN)."
    )
    eval_mrn: Optional[str] = Field(
        default=None,
        description="Alias for patient MRN used in MedAgentBench evaluation."
    )
    clinical_specialties: List[str] = Field(
        default_factory=list,
        description="Specialist personas assigned to the diagnostic panel."
    )
    differential_diagnoses: List[str] = Field(
        default_factory=list,
        description="Candidate differential diagnoses considered by the clinical panel."
    )
    patient_profile: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured EHR context: vitals, lab reports, demographics, imaging notes."
    )
    sol: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Raw benchmark reference solution or solution list."
    )
    task_type: Optional[str] = Field(
        default="diagnostic_dilemma",
        description="Type of clinical task (e.g. diagnostic_dilemma, treatment_planning)."
    )
    initial_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Contextual payload populated into BlackboardState."
    )
    domain: Optional[str] = Field(
        default="clinical_medicine",
        description="Medical domain or department."
    )
    recommended_roles: List[str] = Field(
        default_factory=list,
        description="Recommended agent personas for Scheduler registration."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Raw metadata, benchmark metrics, or resampling markers."
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_medagentbench_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        d = dict(data)

        # 1. Normalize task_id
        task_id = (
            d.get("task_id")
            or d.get("id")
            or d.get("case_id")
            or d.get("eval_MRN")
            or d.get("patient_id")
        )
        if not task_id:
            raise ValueError("Task is missing a valid identifier ('task_id', 'id', 'case_id', or 'eval_MRN').")
        d["task_id"] = str(task_id).strip()

        # 2. Normalize problem_statement
        problem_statement = (
            d.get("problem_statement")
            or d.get("instruction")
            or d.get("prompt")
            or d.get("case_summary")
            or d.get("question")
            or d.get("clinical_vignette")
        )
        if not problem_statement or not str(problem_statement).strip():
            raise ValueError("MedAgentBench task must contain non-empty clinical problem description.")
        d["problem_statement"] = str(problem_statement).strip()

        # 3. Benchmark type
        d["benchmark_type"] = BenchmarkType.MEDAGENTBENCH

        # 4. Patient ID & eval_MRN
        mrn = d.get("patient_id") or d.get("eval_MRN") or d.get("mrn") or d.get("MRN")
        if mrn:
            mrn_str = str(mrn).strip()
            d["patient_id"] = mrn_str
            d["eval_mrn"] = mrn_str
        elif d.get("eval_mrn"):
            d["patient_id"] = str(d["eval_mrn"]).strip()

        # 5. Normalize ground_truth and sol
        sol_raw = d.get("sol") or d.get("solution") or d.get("ground_truth") or d.get("target_diagnosis") or d.get("reference_answer")
        d["sol"] = sol_raw
        if sol_raw is not None:
            if isinstance(sol_raw, list):
                d["ground_truth"] = " | ".join(str(s) for s in sol_raw)
            else:
                d["ground_truth"] = str(sol_raw).strip()
        else:
            d["ground_truth"] = None

        # 6. Patient profile / EHR records
        patient_profile: Dict[str, Any] = {}
        if isinstance(d.get("patient_profile"), dict):
            patient_profile.update(d["patient_profile"])

        raw_context = d.get("context") or d.get("ehr") or d.get("ehr_records") or d.get("patient_data")
        if isinstance(raw_context, dict):
            patient_profile.update(raw_context)
        elif isinstance(raw_context, str) and raw_context.strip():
            patient_profile["clinical_notes"] = raw_context.strip()
        elif isinstance(raw_context, list):
            patient_profile["records"] = raw_context

        d["patient_profile"] = patient_profile

        # 7. Differential diagnoses
        differentials = (
            d.get("differential_diagnoses")
            or d.get("differentials")
            or d.get("candidates")
            or d.get("options")
            or []
        )
        if isinstance(differentials, list):
            d["differential_diagnoses"] = [str(x).strip() for x in differentials if str(x).strip()]
        else:
            d["differential_diagnoses"] = [str(differentials).strip()]

        # 8. Clinical Specialties & Recommended Roles
        specialties = (
            d.get("clinical_specialties")
            or d.get("specialties")
            or d.get("recommended_roles")
            or d.get("panel")
            or d.get("specialist_roles")
        )
        if specialties and isinstance(specialties, list):
            clean_specialties = [str(s).strip() for s in specialties if str(s).strip()]
        else:
            clean_specialties = infer_clinical_specialties(
                text=d["problem_statement"],
                context=patient_profile,
            )

        d["clinical_specialties"] = clean_specialties
        d["recommended_roles"] = clean_specialties

        # 9. Domain
        domain = d.get("domain") or "clinical_medicine"
        d["domain"] = str(domain).strip().lower()

        # 10. Initial Context preparation for BlackboardState
        initial_context = dict(d.get("initial_context") or {})
        initial_context.setdefault("patient_id", d.get("patient_id"))
        initial_context.setdefault("patient_profile", d["patient_profile"])
        initial_context.setdefault("differential_diagnoses", d["differential_diagnoses"])
        initial_context.setdefault("clinical_specialties", d["clinical_specialties"])
        if raw_context is not None and "raw_context" not in initial_context:
            initial_context["raw_context"] = raw_context
        d["initial_context"] = initial_context

        # 11. Metadata
        meta = dict(d.get("metadata") or {})
        meta.setdefault("benchmark_type", BenchmarkType.MEDAGENTBENCH.value)
        meta.setdefault("clinical_specialties", d["clinical_specialties"])
        if d.get("task_type"):
            meta.setdefault("task_type", d["task_type"])
        d["metadata"] = meta

        return d

    def to_blackboard_state(self, session_id: Optional[str] = None) -> BlackboardState:
        """
        Instantiate a live BlackboardState pre-configured for multi-specialist clinical consensus.
        """
        sid = session_id or f"session_medagent_{self.task_id}_{uuid4().hex[:8]}"
        meta = dict(self.metadata)
        meta["benchmark_type"] = self.benchmark_type.value
        meta["clinical_specialties"] = list(self.clinical_specialties)
        meta["recommended_roles"] = list(self.recommended_roles)
        if self.patient_id:
            meta["patient_id"] = self.patient_id
        if self.domain:
            meta["domain"] = self.domain

        return BlackboardState(
            session_id=sid,
            task_id=self.task_id,
            problem_statement=self.problem_statement,
            ground_truth=self.ground_truth,
            initial_context=strip_reference_solutions(dict(self.initial_context)),
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
        # If models map is empty, map each clinical specialist role to a default placeholder
        if not models and self.clinical_specialties:
            for specialty in self.clinical_specialties:
                models[specialty] = "default-clinical-model"

        return TrialConfig(
            benchmark_name=BenchmarkType.MEDAGENTBENCH,
            task_id=self.task_id,
            counterfactual_density=counterfactual_density,
            agent_models=models,
            max_turns=max_turns,
            seed=seed,
        )


class MedAgentBenchAdapter(BaseBenchmarkAdapter[MedAgentBenchTask]):
    """
    Ingestion adapter for the MedAgentBench clinical reasoning benchmark.
    Parses clinical vignettes, extracts patient EHR records, infers specialist roles,
    and supports ablation trial resampling per Section 2.5 of the Project Plan.
    """

    task_cls = MedAgentBenchTask

    def parse_task(self, data: Dict[str, Any]) -> MedAgentBenchTask:
        """
        Parse and validate a dictionary into a typed MedAgentBenchTask contract.
        """
        return MedAgentBenchTask.model_validate(data)

    def infer_specialties(self, text: str, context: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Extract required clinical specialist roles for a given clinical vignette.
        """
        return infer_clinical_specialties(text=text, context=context)

    def resample_for_ablation(
        self,
        tasks: List[MedAgentBenchTask],
        target_count: int,
        seed: int = 42,
        role_permutations: bool = True,
    ) -> List[MedAgentBenchTask]:
        """
        Implement the MedAgentBench resampling and expansion strategy mandated in
        Section 2.5 of the Project Plan:
        - Expands limited published task sets (100–300 tasks) to reach the 500-trials-per-configuration
          ablation requirement (up to 2,000 trials).
        - Generates reproducible, unique task instances with distinct task IDs.
        - Systematically permutes or rotates specialist panel persona seeds and presentation order
          so trial results remain scientifically interpretable.
        """
        if not tasks:
            raise IngestionError("Cannot resample an empty task list.")
        if target_count <= 0:
            raise IngestionError(f"target_count must be positive, got {target_count}")

        rng = random.Random(seed)
        resampled_tasks: List[MedAgentBenchTask] = []

        for i in range(target_count):
            base_task = tasks[i % len(tasks)]
            task_dict = base_task.model_dump()

            replica_id = f"{base_task.task_id}_s{seed}_rep{i}"
            task_dict["task_id"] = replica_id

            meta = dict(task_dict.get("metadata") or {})
            meta["resample_index"] = i
            meta["source_task_id"] = base_task.task_id
            meta["role_seed"] = seed + i

            if role_permutations and base_task.clinical_specialties:
                roles = list(base_task.clinical_specialties)
                # Deterministic rotation based on replica index and seed
                offset = (i + seed) % len(roles)
                permuted_roles = roles[offset:] + roles[:offset]
                task_dict["clinical_specialties"] = permuted_roles
                task_dict["recommended_roles"] = permuted_roles
                meta["role_order"] = permuted_roles

            task_dict["metadata"] = meta
            resampled_tasks.append(self.parse_task(task_dict))

        return resampled_tasks

    def generate_trial_configs(
        self,
        tasks: List[MedAgentBenchTask],
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
