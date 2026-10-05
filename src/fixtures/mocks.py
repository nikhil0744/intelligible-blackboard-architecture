"""
fixtures/mocks.py
Lightweight test doubles for LLM Broker and PEX Agents used in Student 3 test suites.
Keeps testing self-contained and avoids polluting the agents/ and llm_broker/ namespaces owned by Student 2.
"""

from __future__ import annotations

import copy
import uuid
from typing import Any, Dict, List, Optional

from contracts.schemas import BlackboardEntry, PXPTag


class BaseLLMBroker:
    """Abstract base class for test brokers."""

    def generate(self, prompt: str) -> Dict[str, Any]:
        raise NotImplementedError


class MockLLMBroker(BaseLLMBroker):
    """
    Deterministic scripted broker for unit and regression testing.
    Can return scripted response sequences, callable responder responses, or static default dicts.
    """

    def __init__(
        self,
        default_response: Optional[Dict[str, Any]] = None,
        responses: Optional[List[Dict[str, Any]]] = None,
    ):
        self.default_response = default_response or {
            "tag": "RATIFY",
            "prediction": "Default consensus prediction",
            "explanation": "Default consensus explanation with sufficient clinical depth.",
            "confidence": 0.90,
            "evidence_refs": [],
        }
        self.responses = responses or []
        self._index = 0
        self.call_history: List[str] = []

    def enqueue_response(self, response: Dict[str, Any]) -> None:
        """Enqueue a scripted response for sequential return."""
        self.responses.append(response)

    def generate(self, prompt: str) -> Dict[str, Any]:
        self.call_history.append(prompt)
        if self.responses and self._index < len(self.responses):
            res = self.responses[self._index]
            self._index += 1
            return copy.deepcopy(res)
        return copy.deepcopy(self.default_response)


class MockPEXAgent:
    """Lightweight test double for an agent posting to blackboard branches."""

    def __init__(
        self,
        agent_id: str,
        agent_role: str = "generalist",
        broker: Optional[BaseLLMBroker] = None,
    ):
        self.agent_id = agent_id
        self.agent_role = agent_role
        self.broker = broker or MockLLMBroker()

    def generate_turn(
        self,
        blackboard: Any,
        branch_id: str = "main",
        parent_id: Optional[str] = None,
        custom_prompt: Optional[str] = None,
    ) -> BlackboardEntry:
        """Reads branch entries, queries broker, and returns a new BlackboardEntry."""
        branch_entries = blackboard.get_entries(branch_id=branch_id)
        step_number = len(branch_entries) + 1

        prompt = custom_prompt or (
            f"You are {self.agent_id} ({self.agent_role}). "
            f"Current debate history length: {len(branch_entries)}. "
            f"Produce your next PXP prediction and explanation."
        )

        res = self.broker.generate(prompt)

        tag = PXPTag(res.get("tag", PXPTag.RATIFY.value))
        prediction = res.get("prediction", "Default Prediction")
        explanation = res.get("explanation", "Default explanation.")
        confidence = float(res.get("confidence", 0.90))
        evidence_refs = res.get("evidence_refs", [])

        if parent_id is None and branch_entries:
            parent_id = branch_entries[-1].entry_id

        entry = BlackboardEntry(
            entry_id=f"entry_{uuid.uuid4().hex[:8]}",
            parent_id=parent_id,
            branch_id=branch_id,
            agent_id=self.agent_id,
            agent_role=self.agent_role,
            tag=tag,
            prediction=prediction,
            explanation=explanation,
            confidence=confidence,
            evidence_refs=evidence_refs,
            step_number=step_number,
        )
        return blackboard.add_entry(entry)
