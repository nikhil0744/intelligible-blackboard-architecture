"""Ollama backend using only the stdlib HTTP client (POST /api/chat).

Ollama >= 0.5 accepts a JSON schema in `format`, which constrains decoding so
7B models emit valid PEX JSON far more reliably than prompt-only instructions.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List

from ..types import LLMError, LLMRequest, LLMResponse


def _count(v: Any) -> Any:
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


class OllamaBackend:
    name = "ollama"

    def __init__(self, host: str = "http://localhost:11434", timeout_s: float = 120.0, num_ctx: int = 8192):
        self.host = host.rstrip("/")
        self.timeout_s = timeout_s
        self.num_ctx = num_ctx

    # -- helpers -----------------------------------------------------------
    def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        req = urllib.request.Request(
            self.host + path,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise LLMError(f"Ollama HTTP {e.code}: {e.read().decode('utf-8', 'ignore')[:300]}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise LLMError(f"Ollama unreachable at {self.host}: {e}") from e

    def list_models(self) -> List[str]:
        req = urllib.request.Request(self.host + "/api/tags")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError) as e:
            raise LLMError(f"Ollama unreachable at {self.host}: {e}") from e
        return [m["name"] for m in data.get("models", [])]

    # -- LLMBackend --------------------------------------------------------
    def complete(self, request: LLMRequest, model: str) -> LLMResponse:
        options: Dict[str, Any] = {
            "temperature": request.temperature,
            "num_predict": request.max_tokens,
            "num_ctx": self.num_ctx,
        }
        if request.seed is not None:
            options["seed"] = request.seed
        body: Dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump() for m in request.messages],
            "stream": False,
            "options": options,
        }
        if request.json_schema is not None:
            body["format"] = request.json_schema

        t0 = time.perf_counter()
        data = self._post("/api/chat", body)
        latency = (time.perf_counter() - t0) * 1000
        if "error" in data:
            raise LLMError(f"Ollama error: {data['error']}")
        return LLMResponse(
            text=data.get("message", {}).get("content", ""),
            model=model,
            backend=self.name,
            # Ollama omits these counts in some cases (e.g. a fully cached prompt): report unknown, not 0.
            prompt_tokens=_count(data.get("prompt_eval_count")),
            completion_tokens=_count(data.get("eval_count")),
            latency_ms=latency,
        )
