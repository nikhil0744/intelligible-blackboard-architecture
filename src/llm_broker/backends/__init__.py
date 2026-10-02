"""Pluggable inference backends. Each implements `LLMBackend.complete(request, model)`."""

from .base import LLMBackend
from .mock import MockBackend
from .ollama import OllamaBackend

__all__ = ["LLMBackend", "MockBackend", "OllamaBackend", "LiteLLMBackend"]


def __getattr__(name):  # lazy: litellm is an optional dependency
    if name == "LiteLLMBackend":
        from .litellm_backend import LiteLLMBackend

        return LiteLLMBackend
    raise AttributeError(name)
