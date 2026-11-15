"""Shared single-task execution. Notebook and CLI both call this real scheduler path."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

from agents.base import AgentTurnError
from agents.board_client import InMemoryBoard
from contracts.schemas import BlackboardStatus, PXPTag
from llm_broker.budget import BudgetExceeded, CallBudget, budget_scope
from llm_broker.ledger import usage_scope
from prompts.validation import normalize_claim
from scheduler.engine import DeterministicScheduler
from scheduler.policies import RoundRobinPolicy

from .actions import BenchmarkAgent, ToolAction, active_root, root_id
from .data_tools import DataTools


@dataclass(frozen=True)
class Limits:
    max_turns: int = 15
    max_tool_rounds: int = 24
    max_calls: int = 90
    task_timeout_s: float = 1200
    tool_timeout_s: float = 15
    run_timeout_s: float = 3600

    def __post_init__(self):
        if min(asdict(self).values()) <= 0:
            raise ValueError("all execution limits must be positive")


ROSTER = (("analyst", "data_analyst"), ("critic", "data_engineer"), ("reasoner", "general_reasoner"))


class BenchmarkScheduler(DeterministicScheduler):
    """Canonical scheduler with explicit three-person votes on the current revision."""

    def _check_consensus(self, snapshot):
        root = active_root(snapshot)
        if root is None:
            return False
        votes = set()
        final = False
        for c in snapshot.contributions[root.turn_index:]:
            if c.tag in {PXPTag.REFUTE, PXPTag.REJECT}:
                votes.clear()
            if root_id(snapshot, c.contribution_id) == root.contribution_id:
                if c.metadata.get("final_submission"):
                    final = True
                if c.tag == PXPTag.RATIFY and normalize_claim(c.payload.prediction.claim) == normalize_claim(root.payload.prediction.claim):
                    votes.add(c.agent_id)
        return final and set(self.registered_agents) <= votes


class ToolLimit(RuntimeError):
    pass


class SessionRunner:
    """One controller owns tools and commits; reference data is not an argument here."""

    def __init__(self, broker, settings, limits: Limits, seed=0, observer=None):
        self.broker, self.settings, self.limits, self.seed = broker, settings, limits, seed
        self.observer = observer or (lambda event: None)

    def execute(self, task, input_dir: Path, attempt):
        started = time.monotonic()
        sid = attempt.identity["session_id"]
        aid = attempt.identity["attempt_id"]
        board = InMemoryBoard()
        board.create_session(task.task_id, task.question, {"permitted_files": list(task.files)}, sid)
        scheduler = BenchmarkScheduler(sid, board, policy=RoundRobinPolicy(), deadlock_window=2, consensus_threshold=3)
        budget = CallBudget(max_attempts=self.limits.max_calls, timeout_s=self.limits.task_timeout_s)
        tools = DataTools(input_dir, task.files, timeout_s=self.limits.tool_timeout_s)
        observations = []
        tool_rounds = 0

        def event(kind, **payload):
            attempt.event(kind, **payload)
            self.observer({"kind": kind, "task_id": task.task_id, **payload})

        def handler(agent):
            def turn(snapshot, scheduled):
                nonlocal tool_rounds
                while True:
                    budget.check()
                    snapshot = board.get_snapshot(sid)
                    action = agent.action(snapshot)
                    if not isinstance(action, ToolAction):
                        return action
                    if tool_rounds >= self.limits.max_tool_rounds:
                        raise ToolLimit("tool_limit")
                    request_id = uuid4().hex
                    tool_rounds += 1
                    event("tool_requested", request_id=request_id, agent_id=agent.agent_id,
                          name=action.name, arguments=action.arguments, board_version=snapshot.version,
                          turn_index=len(snapshot.contributions), branch="main")
                    tools.timeout_s = min(self.limits.tool_timeout_s, budget.remaining())
                    result = tools.execute(request_id, action.name, action.arguments)
                    event("tool_observed", request_id=request_id, agent_id=agent.agent_id, result=result)
                    # Keep recent observations inside the ordinary prompt budget. Complete
                    # observations stay in the durable event log, never silently in a cache.
                    observation = {"request_id": request_id, "agent_id": agent.agent_id,
                                   "name": action.name, "arguments": action.arguments,
                                   "result": result}
                    encoded = json.dumps(observation, ensure_ascii=False)
                    if len(encoded) > 1600:
                        observation["result"] = {"excerpt": json.dumps(result, ensure_ascii=False)[:1200],
                                                 "truncated": True, "hint": "Narrow query/select/limit to inspect the needed data."}
                    if len(json.dumps(observation["arguments"])) > 400:
                        observation["arguments"] = {"excerpt": json.dumps(action.arguments)[:400], "truncated": True}
                    observations.append(observation)
                    recent = observations[-3:]
                    context = {"permitted_files": list(task.files), "latest_tool_observations": recent}
                    while len(json.dumps(context, ensure_ascii=False)) > 5500 and len(recent) > 1:
                        recent.pop(0)
                    board.update_context(sid, context)
            return turn

        panel = [BenchmarkAgent(name, persona, self.broker, model=self.settings.model,
                                seed=self.seed, counterfactual_capable=False, **self.settings.agent_kwargs())
                 for name, persona in ROSTER]
        for agent in panel:
            scheduler.register_agent(agent.agent_id, agent.role, handler(agent))
        old_hook = self.broker.response_hook

        def response_hook(request, response):
            event("model_response", agent_id=request.agent_id,
                  messages=[m.model_dump() for m in request.messages], response=response.text,
                  is_repair=request.metadata.get("is_repair", False))
            if old_hook:
                old_hook(request, response)

        error = None
        status = "turn_limit"
        self.broker.response_hook = response_hook
        self.broker.ledger.attach_file(attempt.path / "calls.jsonl")
        try:
            event("session_started", task=task.model_dump(mode="json"), recovery_mode="disabled",
                  capable_agents=[], roster=[agent.spec() for agent in panel])
            with usage_scope(trial_id=aid, session_id=sid, phase="deliberation"), budget_scope(budget):
                for _ in range(self.limits.max_turns):
                    budget.check()
                    contribution = scheduler.step()
                    if contribution is None:
                        status = "queue_empty"
                        break
                    event("contribution", contribution=contribution.model_dump(mode="json"))
                    snapshot = board.get_snapshot(sid)
                    if snapshot.status == BlackboardStatus.CONSENSUS:
                        status = "consensus"
                        break
                    if snapshot.status == BlackboardStatus.DEADLOCK:
                        status = "deadlock"
                        event("deadlock_detected", recovery_mode="disabled")
                        break
        except AgentTurnError as exc:
            error = exc.to_record()
            # PEXAgent wraps a budget denial from the broker in AgentTurnError.
            reason = getattr(exc.__cause__, "reason", None)
            status = reason or exc.kind
        except BudgetExceeded as exc:
            status, error = exc.reason, {"error": str(exc)}
        except ToolLimit as exc:
            status, error = "tool_limit", {"error": str(exc)}
        except KeyboardInterrupt:
            status, error = "interrupted", {"error": "Stopped by user; resume restarts this task as a new attempt."}
        finally:
            self.broker.response_hook = old_hook
            self.broker.ledger.attach_file(None)
        snapshot = board.get_snapshot(sid)
        root = active_root(snapshot)
        output = None
        if root:
            submissions = [c for c in snapshot.contributions if c.metadata.get("final_submission")
                           and root_id(snapshot, c.contribution_id) == root.contribution_id]
            if submissions:
                output = submissions[-1].metadata["final_output"]
        event("session_stopped", status=status, consensus=snapshot.status == BlackboardStatus.CONSENSUS,
              turns=len(snapshot.contributions), tool_rounds=tool_rounds)
        usage = self.broker.trial_usage(aid)
        return {"status": status, "error": error, "output": output,
                "consensus": snapshot.status == BlackboardStatus.CONSENSUS,
                "turns": len(snapshot.contributions), "tool_rounds": tool_rounds,
                "recovery": {"mode": "disabled", "calls": 0, "applied": False, "confirmed": False},
                "intelligibility": "unassessed", "elapsed_s": time.monotonic() - started,
                "usage": {**usage.model_dump(), "total_tokens": usage.total_tokens}}
