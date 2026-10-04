"""Local LLM broker (Student 2): multi-threaded calling interface + resource guard."""

from .broker import ModelBroker
from .config import DEFAULT_MODEL, InferenceSettings, build_broker, build_broker_from_env
from .ledger import PHASES, UsageLedger, usage_scope
from .types import CallRecord, ChatMessage, LLMError, LLMRequest, LLMResponse, UsageStats, UsageSummary

__all__ = [
    "ModelBroker",
    "build_broker",
    "build_broker_from_env",
    "InferenceSettings",
    "DEFAULT_MODEL",
    "ChatMessage",
    "LLMError",
    "LLMRequest",
    "LLMResponse",
    "UsageStats",
    "UsageSummary",
    "CallRecord",
    "UsageLedger",
    "usage_scope",
    "PHASES",
]
