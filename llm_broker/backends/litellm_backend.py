"""LiteLLM backend: one interface for Ollama, vLLM (OpenAI-compatible) or a hosted API.

Model names follow LiteLLM conventions, e.g. "ollama_chat/qwen2.5:7b-instruct-q4_K_M"
or "hosted_vllm/Qwen/Qwen2.5-7B-Instruct" (set api_base).
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from ..types import LLMError, LLMRequest, LLMResponse


class LiteLLMBackend:
    name = "litellm"

    def __init__(self, api_base: Optional[str] = None, timeout_s: float = 120.0):
        try:
            import litellm  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise LLMError("litellm not installed: pip install -r requirements-s2.txt") from e
        self.api_base = api_base
        self.timeout_s = timeout_s

    def complete(self, request: LLMRequest, model: str) -> LLMResponse:
        import litellm

        kwargs: Dict[str, Any] = dict(
            model=model,
            messages=[m.model_dump() for m in request.messages],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            timeout=self.timeout_s,
        )
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if request.seed is not None:
            kwargs["seed"] = request.seed
        if request.json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "pex_decision", "schema": request.json_schema},
            }
        t0 = time.perf_counter()
        try:
            resp = litellm.completion(**kwargs)
        except Exception as e:  # litellm raises many provider-specific types
            raise LLMError(f"LiteLLM call failed: {e}") from e
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            model=model,
            backend=self.name,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
