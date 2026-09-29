"""
counterfactual package exports.
"""

from .attribution import AttributionResult, DeadlockAttributionEngine
from .scoring import (
    BranchScoreDetails,
    CreditAssignmentScorer,
    PlateauDetector,
)
from .simulator import (
    CounterfactualSimulator,
    SimulatedCandidate,
    SimulationResult,
    SimulationStatus,
)

__all__ = [
    "DeadlockAttributionEngine",
    "AttributionResult",
    "CounterfactualSimulator",
    "SimulationStatus",
    "SimulatedCandidate",
    "SimulationResult",
    "BranchScoreDetails",
    "PlateauDetector",
    "CreditAssignmentScorer",
]

