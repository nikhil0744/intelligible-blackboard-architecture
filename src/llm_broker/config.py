"""Inference settings and broker construction from environment variables (.env supported).

`InferenceSettings` holds the pilot defaults agreed for the study. Change them only
during development / pilot calibration, then freeze them for the final runs; the
effective values are written into every run manifest.

There is no fallback path: the configured backend and model are used or the call
fails. Nothing here switches model, pulls a model, or moves to a paid API by itself.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Dict

from .broker import ModelBroker

DEFAULT_MODEL = "qwen2.5:7b-instruct-q4_K_M"


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


@dataclass(frozen=True)
class InferenceSettings:
    backend: str = "ollama"
    model: str = DEFAULT_MODEL
    temperature: float = 0.3
    num_ctx: int = 8192  # context window in tokens
    max_tokens: int = 768  # maximum generated response
    max_repairs: int = 2  # repair responses after an invalid first response
    timeout_s: float = 120.0  # per request
    max_concurrency: int = 1  # one loaded model, one request at a time
    max_retries: int = 2  # backend retries after a transport/server error
    ollama_host: str = "http://localhost:11434"
    api_base: str = ""  # litellm only

    @classmethod
    def from_env(cls, **overrides: Any) -> "InferenceSettings":
        """Read LLM_* / OLLAMA_HOST variables. `overrides` accepts field names or the legacy
        lower-cased variable names (llm_backend, llm_model, ...); None values are ignored."""
        _load_dotenv()
        d = cls()
        env_names = {
            "backend": "LLM_BACKEND",
            "model": "LLM_MODEL",
            "temperature": "LLM_TEMPERATURE",
            "num_ctx": "LLM_NUM_CTX",
            "max_tokens": "LLM_MAX_TOKENS",
            "max_repairs": "LLM_MAX_REPAIRS",
            "timeout_s": "LLM_TIMEOUT_S",
            "max_concurrency": "LLM_MAX_CONCURRENCY",
            "max_retries": "LLM_MAX_RETRIES",
            "ollama_host": "OLLAMA_HOST",
            "api_base": "LLM_API_BASE",
        }
        values: Dict[str, Any] = {}
        for name, var in env_names.items():
            raw = overrides.get(name)
            if raw is None:
                raw = overrides.get(var.lower())
            if raw is None:
                raw = os.getenv(var)
            if raw is None or raw == "":
                continue
            values[name] = type(getattr(d, name))(raw)
        values["backend"] = str(values.get("backend", d.backend)).lower()
        return cls(**values)

    def agent_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments for PEXAgent / build_panel so agents decode with these settings."""
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_parse_retries": self.max_repairs,
            "context_tokens": self.num_ctx,
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_broker(settings: InferenceSettings) -> ModelBroker:
    if settings.backend == "ollama":
        from .backends.ollama import OllamaBackend

        backend = OllamaBackend(host=settings.ollama_host, timeout_s=settings.timeout_s, num_ctx=settings.num_ctx)
    elif settings.backend == "litellm":
        from .backends.litellm_backend import LiteLLMBackend

        backend = LiteLLMBackend(api_base=settings.api_base or None, timeout_s=settings.timeout_s)
    elif settings.backend == "mock":
        from .backends.mock import MockBackend

        backend = MockBackend()
    else:
        raise ValueError(f"Unknown LLM_BACKEND={settings.backend!r} (ollama|litellm|mock)")

    broker = ModelBroker(
        backend=backend,
        default_model=settings.model,
        max_concurrency=settings.max_concurrency,
        max_retries=settings.max_retries,
    )
    broker.settings = settings
    return broker


def build_broker_from_env(**overrides: Any) -> ModelBroker:
    return build_broker(InferenceSettings.from_env(**overrides))
