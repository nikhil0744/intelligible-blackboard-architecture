"""Export JSON Schema stubs for S2, S3, and S4 cross-module integration.

Usage:
    python -m contracts.export
"""

import json
from pathlib import Path
from contracts.schemas import (
    AgentContribution,
    BlackboardSnapshot,
    DeadlockNotification,
    LockAcquireRequest,
    LockAcquireResponse,
    PEXPayload,
    Prediction,
    Explanation,
    PXPTag,
    ReviseProposal,
    StateWriteRequest,
    TelemetryEvent,
    TrialConfig,
    TrialResult,
)

EXPORT_MODELS = {
    "PEXPayload": PEXPayload,
    "Prediction": Prediction,
    "Explanation": Explanation,
    "AgentContribution": AgentContribution,
    "BlackboardSnapshot": BlackboardSnapshot,
    "DeadlockNotification": DeadlockNotification,
    "LockAcquireRequest": LockAcquireRequest,
    "LockAcquireResponse": LockAcquireResponse,
    "StateWriteRequest": StateWriteRequest,
    "ReviseProposal": ReviseProposal,
    "TelemetryEvent": TelemetryEvent,
    "TrialConfig": TrialConfig,
    "TrialResult": TrialResult,
}


def export_schemas(output_dir: Path | None = None) -> None:
    if output_dir is None:
        output_dir = Path(__file__).parent / "stubs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting Pydantic v2 JSON schemas to {output_dir.resolve()}...")
    for name, model in EXPORT_MODELS.items():
        schema = model.model_json_schema()
        schema_file = output_dir / f"{name}.schema.json"
        with open(schema_file, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)
        print(f"  [OK] Exported {schema_file.name}")

    print("\nAll schema stubs successfully exported!")


if __name__ == "__main__":
    export_schemas()
