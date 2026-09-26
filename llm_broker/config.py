"""Build a ModelBroker from environment variables (.env supported)."""

from __future__ import annotations

import os

from .broker import ModelBroker


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def build_broker_from_env(**overrides) -> ModelBroker:
    _load_dotenv()
    env = lambda k, d: overrides.get(k.lower(), os.getenv(k, d))  # noqa: E731
    backend_name = str(env("LLM_BACKEND", "ollama")).lower()
    timeout = float(env("LLM_TIMEOUT_S", 120))

    if backend_name == "ollama":
        from .backends.ollama import OllamaBackend

        backend = OllamaBackend(
            host=env("OLLAMA_HOST", "http://localhost:11434"),
            timeout_s=timeout,
            num_ctx=int(env("LLM_NUM_CTX", 8192)),
        )
    elif backend_name == "litellm":
        from .backends.litellm_backend import LiteLLMBackend

        backend = LiteLLMBackend(api_base=env("LLM_API_BASE", None), timeout_s=timeout)
    elif backend_name == "mock":
        from .backends.mock import MockBackend

        backend = MockBackend()
    else:
        raise ValueError(f"Unknown LLM_BACKEND={backend_name!r} (ollama|litellm|mock)")

    return ModelBroker(
        backend=backend,
        default_model=env("LLM_MODEL", "qwen2.5:7b-instruct-q4_K_M"),
        max_concurrency=int(env("LLM_MAX_CONCURRENCY", 2)),
        max_retries=int(env("LLM_MAX_RETRIES", 2)),
    )
