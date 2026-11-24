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
from scheduler.engine import DeterministicScheduler
from scheduler.policies import RoundRobinPolicy

from .actions import BenchmarkAgent, ToolAction, active_root, benchmark_claim_key, root_id
from .data_tools import DataTools


@dataclass(frozen=True)
class Limits:
    max_turns: int = 15
    max_tool_rounds: int = 24
    max_calls: int = 90
    task_timeout_s: float = 1200
    tool_timeout_s: float = 15
    run_timeout_s: float = 3600
    max_tools_per_turn: int = 8
    recovery_enabled: bool = True
    max_recovery_turns: int = 6
    max_recovery_tool_rounds: int = 8

    def __post_init__(self):
        if any(value <= 0 for value in asdict(self).values() if not isinstance(value, bool)):
            raise ValueError("all execution limits must be positive")


ROSTER = (("analyst", "data_analyst"), ("critic", "data_engineer"), ("reasoner", "general_reasoner"))


class InsufficientEvidence(RuntimeError):
    pass


def _query_signature(args):
    """Recognize equivalent query requests when choosing a recovery alternative."""
    def value_key(value):
        if isinstance(value, str):
            return value.strip().casefold()
        if isinstance(value, list):
            return sorted(json.dumps(value_key(v), sort_keys=True) for v in value)
        return value

    try:
        encoded_filters = sorted(json.dumps({key: value_key(value) for key, value in entry.items()},
                                            sort_keys=True) for entry in args.get("filters", []))
        aggregate = args.get("aggregate") or {}
        grouping = args.get("group_by", [])
        row_options = [args.get("sort"), args.get("offset", 0), args.get("limit", 10)] if not aggregate or grouping else []
        first_per = args.get("first_per")
        return json.dumps([args.get("file"), args.get("header"), encoded_filters,
                           {key: value_key(value) for key, value in aggregate.items()},
                           [value_key(col) for col in grouping],
                           [value_key(col) for col in args.get("select", [])] if not aggregate else [],
                           args.get("derive"), first_per, args.get("sort") if first_per else None, row_options],
                          sort_keys=True)
    except (AttributeError, TypeError):
        # The tool executor will return validation diagnostics for malformed arguments.
        return json.dumps(args, sort_keys=True)


class BenchmarkScheduler(DeterministicScheduler):
    """Canonical scheduler with explicit three-person votes on the current revision."""

    def __init__(self, *args, answer_type="numeric_exact", **kwargs):
        super().__init__(*args, **kwargs)
        self.answer_type = answer_type

    def vote_state(self, snapshot):
        root = active_root(snapshot)
        votes = set()
        final = False
        if root:
            claim = benchmark_claim_key(root.payload.prediction.claim, self.answer_type)
            for c in snapshot.contributions[root.turn_index:]:
                if root_id(snapshot, c.contribution_id) != root.contribution_id:
                    continue
                if c.tag in {PXPTag.REFUTE, PXPTag.REJECT}:
                    votes.clear()
                if c.metadata.get("final_submission"):
                    final = True
                if c.tag == PXPTag.RATIFY and benchmark_claim_key(c.payload.prediction.claim, self.answer_type) == claim:
                    votes.add(c.agent_id)
        return {
            "claim_id": root.contribution_id if root else None,
            "final_submission": final,
            "ratified_agent_ids": [aid for aid in self.registered_agents if aid in votes],
            "pending_agent_ids": [aid for aid in self.registered_agents if aid not in votes],
        }

    def _check_consensus(self, snapshot):
        state = self.vote_state(snapshot)
        return state["final_submission"] and not state["pending_agent_ids"]


class SessionRunner:
    """One controller owns tools and commits; reference data is not an argument here."""

    def __init__(self, broker, settings, limits: Limits, seed=0, observer=None):
        self.broker, self.settings, self.limits, self.seed = broker, settings, limits, seed
        self.observer = observer or (lambda event: None)

    def execute(self, task, input_dir: Path, attempt):
        started = time.monotonic()
        sid = attempt.identity["session_id"]
        aid = attempt.identity["attempt_id"]
        budget = CallBudget(max_attempts=self.limits.max_calls, timeout_s=self.limits.task_timeout_s)
        tools = DataTools(input_dir, task.files, timeout_s=self.limits.tool_timeout_s, question=task.question)
        observations = []
        tool_rounds = 0
        latest_query = None
        novel_query = None
        evidence_by_file = {}
        strict_scope = task.scope_mode == "strict"
        multi_file = len(task.files) > 1
        distinct_queries = set()
        inspected_headers: dict[str, int] = {}
        branch = "main"
        prior_queries = set()
        recovery_tool_start = 0
        recovery_context = None
        recovery_mode = "bounded_evidence_backtrack" if self.limits.recovery_enabled else "disabled"

        def event(kind, **payload):
            payload.setdefault("branch", branch)
            attempt.event(kind, **payload)
            self.observer({"kind": kind, "task_id": task.task_id, **payload})

        def has_evidence():
            return set(task.files) <= evidence_by_file.keys() and (branch == "main" or novel_query is not None)

        def handler(agent):
            def turn(snapshot, scheduled):
                nonlocal tool_rounds, latest_query, novel_query
                requests = set()
                reminder = None
                while True:
                    budget.check()
                    snapshot = board.get_snapshot(sid)
                    vote_state = scheduler.vote_state(snapshot)
                    snapshot.initial_context = {**snapshot.initial_context, "votes": vote_state}
                    if multi_file:
                        snapshot.initial_context["evidence_pending_files"] = [
                            file for file in task.files if file not in evidence_by_file
                        ]
                    if tool_rounds >= self.limits.max_tool_rounds:
                        stop_reason = "session_tool_budget"
                    elif branch == "recovery" and tool_rounds - recovery_tool_start >= self.limits.max_recovery_tool_rounds:
                        stop_reason = "recovery_tool_budget"
                    elif len(requests) >= self.limits.max_tools_per_turn:
                        stop_reason = "turn_tool_budget"
                    else:
                        stop_reason = None
                    if stop_reason:
                        event("tools_closed", agent_id=agent.agent_id, reason=stop_reason)
                    opening = not snapshot.contributions
                    if opening and stop_reason:
                        if not has_evidence():
                            raise InsufficientEvidence("Tool budget exhausted before collecting the required CSV evidence")
                        return agent.action(snapshot, phase="submit")
                    if opening:
                        phase = "opening" if has_evidence() else "collect"
                    else:
                        phase = "review_closed" if stop_reason else "review"
                    action = agent.action(snapshot, reminder, phase=phase)
                    if not isinstance(action, ToolAction):
                        return action
                    args = action.arguments.copy()
                    file = args.get("file")
                    inherited_header = (action.name == "query" and "header" not in args
                                        and isinstance(file, str) and file in inspected_headers)
                    if inherited_header:
                        args["header"] = inspected_headers[file]
                    request = (action.name, json.dumps(args, sort_keys=True))
                    query = _query_signature(args) if action.name == "query" else None
                    if branch == "recovery" and opening and not multi_file and query in prior_queries:
                        event("tool_request_blocked", agent_id=agent.agent_id, name=action.name,
                              arguments=args, reason="prior_branch_query")
                        if has_evidence():
                            return agent.action(snapshot, "Use the new evidence and submit your answer.", phase="submit")
                        if reminder:
                            raise InsufficientEvidence("Recovery repeatedly requested the same prior-branch query")
                        reminder = "That query already ran in the prior branch. Choose a different scope, measure, or raw row query."
                        continue
                    if request in requests:
                        event("tool_request_blocked", agent_id=agent.agent_id, name=action.name,
                              arguments=args, reason="duplicate_in_turn")
                        if opening and has_evidence():
                            return agent.action(snapshot, "You already have usable data. Check its scope and submit.",
                                                phase="submit")
                        if not opening:
                            return agent.action(snapshot, "This tool request already ran in this turn. Use the saved observation.",
                                                phase="review_closed")
                        # A failed investigation needs an alternative tool request, not an invented claim.
                        if reminder:
                            raise InsufficientEvidence("Repeated tool requests produced no usable CSV query")
                        reminder = "Your last request was a duplicate. Use the observation and change your tool or arguments."
                        continue
                    requests.add(request)
                    reminder = None
                    request_id = uuid4().hex
                    tool_rounds += 1
                    event("tool_requested", request_id=request_id, agent_id=agent.agent_id,
                          name=action.name, arguments=args, header_inherited=inherited_header, board_version=snapshot.version,
                          turn_index=len(snapshot.contributions))
                    tools.timeout_s = min(self.limits.tool_timeout_s, budget.remaining())
                    result = tools.execute(request_id, action.name, args)
                    event("tool_observed", request_id=request_id, agent_id=agent.agent_id, result=result)
                    if branch == "main" and query is not None:
                        prior_queries.add(query)
                    data = result.get("data")
                    scope_issues = data.get("scope_issues", []) if isinstance(data, dict) else []
                    if action.name == "inspect" and isinstance(file, str) and isinstance(data, dict) and "header" in data:
                        inspected_headers[file] = data["header"]
                    # Keep recent observations inside the ordinary prompt budget. Complete
                    # observations stay in the durable event log, never silently in a cache.
                    observation = {"request_id": request_id, "agent_id": agent.agent_id,
                                   "name": action.name, "arguments": args,
                                   "result": result}
                    encoded = json.dumps(observation, ensure_ascii=False)
                    if len(encoded) > 1600:
                        data = result.get("data")
                        if action.name == "inspect" and isinstance(data, dict):
                            observation["result"] = {
                                "ok": result["ok"], "header": data.get("header"), "columns": data.get("columns", [])[:24],
                                "suggested_header": data.get("suggested_header"),
                                "header_candidates": data.get("header_candidates", [])[:8],
                                "raw_first_lines": [line[:140] for line in data.get("raw_first_lines", [])[:6]],
                                "truncated": True,
                            }
                        else:
                            observation["result"] = {"excerpt": json.dumps(result, ensure_ascii=False)[:1200],
                                                     "truncated": True, "hint": "Narrow query/select/limit to inspect the needed data."}
                            if scope_issues:
                                observation["result"]["scope_issues"] = scope_issues
                    if len(json.dumps(observation["arguments"])) > 400:
                        observation["arguments"] = {"excerpt": json.dumps(args)[:400], "truncated": True}
                    observations.append(observation)
                    data = result.get("data")
                    if scope_issues:
                        event("query_scope_mismatch", agent_id=agent.agent_id, request_id=request_id,
                              issues=scope_issues, blocking=strict_scope)
                    if (action.name == "query" and result.get("ok") and isinstance(data, dict)
                            and (not strict_scope or not scope_issues)
                            and data.get("matched_rows", 0) > 0
                            and (data.get("value") is not None or data.get("rows"))):
                        had_evidence = has_evidence()
                        latest_query = observation
                        evidence_by_file[file] = observation
                        if branch == "recovery" and query not in prior_queries:
                            novel_query = observation
                        if opening and not had_evidence and has_evidence():
                            event("evidence_ready", agent_id=agent.agent_id, request_id=request_id,
                                  evidence_request_ids=[evidence_by_file[name]["request_id"] for name in task.files])
                        distinct_queries.add(request)
                    recent = observations[-3:]
                    context = {"permitted_files": list(task.files), "answer_type": task.answer_type}
                    if multi_file:
                        context["query_evidence_by_file"] = evidence_by_file
                        recent = [observation for observation in recent if observation not in evidence_by_file.values()]
                    elif latest_query is not None and latest_query not in recent:
                        context["latest_successful_query"] = latest_query
                    if recovery_context is not None:
                        context["recovery"] = recovery_context
                    context["latest_tool_observations"] = recent
                    while len(json.dumps(context, ensure_ascii=False)) > 5500 and recent:
                        recent.pop(0)
                    board.update_context(sid, context)
            return turn

        panel = [BenchmarkAgent(name, persona, self.broker, model=self.settings.model,
                                seed=self.seed, counterfactual_capable=False, answer_type=task.answer_type,
                                **self.settings.agent_kwargs())
                 for name, persona in ROSTER]

        def new_board(context):
            store = InMemoryBoard()
            store.create_session(task.task_id, task.question, context, sid)
            engine = BenchmarkScheduler(sid, store, policy=RoundRobinPolicy(), deadlock_window=2,
                                        consensus_threshold=3, answer_type=task.answer_type)
            for agent in panel:
                engine.register_agent(agent.agent_id, agent.role, handler(agent))
            return store, engine

        board, scheduler = new_board({"permitted_files": list(task.files), "answer_type": task.answer_type})
        old_hook = self.broker.response_hook

        def response_hook(request, response):
            event("model_response", agent_id=request.agent_id,
                  messages=[m.model_dump() for m in request.messages], response=response.text,
                  is_repair=request.metadata.get("is_repair", False))
            if old_hook:
                old_hook(request, response)

        total_turns = 0
        recovery = {"mode": recovery_mode, "attempts": 0, "calls": 0, "applied": False, "confirmed": False}

        def run_branch(turn_limit):
            nonlocal total_turns
            turn_signatures = []
            try:
                for _ in range(turn_limit):
                    budget.check()
                    contribution = scheduler.step()
                    if contribution is None:
                        return "queue_empty", None
                    total_turns += 1
                    snapshot = board.get_snapshot(sid)
                    event("contribution", contribution=contribution.model_dump(mode="json"),
                          votes=scheduler.vote_state(snapshot))
                    if branch == "recovery" and contribution.tag == PXPTag.REVISE and contribution.metadata.get("final_submission"):
                        event("recovery_candidate", contribution_id=contribution.contribution_id,
                              claim=contribution.payload.prediction.claim,
                              evidence_request_id=novel_query["request_id"],
                              evidence_request_ids=[evidence_by_file[name]["request_id"] for name in task.files],
                              source_claim=recovery_context["source_claim"])
                    if snapshot.status == BlackboardStatus.CONSENSUS:
                        return "consensus", None
                    if snapshot.status == BlackboardStatus.DEADLOCK:
                        event("deadlock_detected", recovery_mode=recovery_mode)
                        return "deadlock", None
                    turn_signatures.append((contribution.agent_id, contribution.tag,
                                            benchmark_claim_key(contribution.payload.prediction.claim, task.answer_type),
                                            len(distinct_queries)))
                    cycle = len(panel)
                    if (len(turn_signatures) >= 2 * cycle
                            and turn_signatures[-2 * cycle:-cycle] == turn_signatures[-cycle:]
                            and any(item[1] in {PXPTag.REFUTE, PXPTag.REJECT} for item in turn_signatures[-cycle:])):
                        event("discussion_stalled", reason="Repeated disagreement without new CSV queries")
                        return "stalled", None
            except AgentTurnError as exc:
                # PEXAgent wraps budget denials raised by the broker.
                return getattr(exc.__cause__, "reason", None) or exc.kind, exc.to_record()
            except BudgetExceeded as exc:
                return exc.reason, {"error": str(exc)}
            except InsufficientEvidence as exc:
                return "insufficient_evidence", {"error": str(exc)}
            except KeyboardInterrupt:
                return "interrupted", {"error": "Stopped by user; resume restarts this task as a new attempt."}
            return "turn_limit", None

        error = None
        self.broker.response_hook = response_hook
        self.broker.ledger.attach_file(attempt.path / "calls.jsonl")
        try:
            event("session_started", task=task.model_dump(mode="json"), recovery_mode=recovery_mode,
                  roster=[agent.spec() for agent in panel])
            with budget_scope(budget), usage_scope(trial_id=aid, session_id=sid):
                status, error = run_branch(self.limits.max_turns)
                if self.limits.recovery_enabled and status in {"insufficient_evidence", "deadlock", "stalled"}:
                    recovery_turns = min(self.limits.max_recovery_turns, self.limits.max_turns - total_turns)
                    if (recovery_turns < len(panel) + 1 or tool_rounds >= self.limits.max_tool_rounds
                            or budget.attempts >= self.limits.max_calls or budget.remaining() <= 0):
                        event("recovery_skipped", reason="Insufficient shared budget for a new candidate and fresh votes")
                    else:
                        main_board, main_scheduler = board, scheduler
                        original = main_board.get_snapshot(sid)
                        source_root = active_root(original)
                        recovery_context = {
                            "reason": status,
                            "source_claim": source_root.payload.prediction.claim[:200] if source_root else None,
                            "challenges": [post.payload.explanation.rationale[:240]
                                           for post in original.contributions if post.tag in {PXPTag.REFUTE, PXPTag.REJECT}][-3:],
                            "instruction": "This branch has rewound before the opening claim. Check a different CSV query before submitting.",
                        }
                        if latest_query is not None:
                            recovery_context["prior_query"] = latest_query
                        if multi_file:
                            recovery_context["instruction"] += (
                                " Recheck each file. At least one supporting query must differ from the prior branch."
                            )
                        recovery_context["recent_prior_observations"] = observations[-2:]
                        observations = []
                        latest_query = None
                        novel_query = None
                        evidence_by_file = {}
                        branch = "recovery"
                        recovery_tool_start = tool_rounds
                        recovery.update(attempts=1, reason=status)
                        event("recovery_started", reason=status, parent_branch="main", checkpoint="before_opening_claim",
                              source_contribution_id=source_root.contribution_id if source_root else None,
                              source_claim=recovery_context["source_claim"])
                        board, scheduler = new_board({"permitted_files": list(task.files), "answer_type": task.answer_type,
                                                      "recovery": recovery_context})
                        with usage_scope(phase="candidate_replay"):
                            recovered_status, recovered_error = run_branch(recovery_turns)
                        candidate = active_root(board.get_snapshot(sid))
                        confirmed = recovered_status == "consensus"
                        recovery.update(
                            status=recovered_status, confirmed=confirmed, applied=confirmed,
                            calls=self.broker.ledger.summary(trial_id=aid, phase="candidate_replay").calls,
                            tool_rounds=tool_rounds - recovery_tool_start,
                            candidate_id=candidate.contribution_id if candidate else None,
                            error=recovered_error,
                        )
                        event("recovery_finished", status=recovered_status, applied=confirmed, confirmed=confirmed,
                              candidate_id=recovery["candidate_id"], votes=scheduler.vote_state(board.get_snapshot(sid)))
                        if confirmed or recovered_status == "interrupted":
                            status, error = recovered_status, recovered_error
                        if not confirmed:
                            board = main_board
                            scheduler = main_scheduler
                            branch = "main"
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
              turns=total_turns, tool_rounds=tool_rounds, votes=scheduler.vote_state(snapshot))
        usage = self.broker.trial_usage(aid)
        return {"status": status, "error": error, "output": output,
                "consensus": snapshot.status == BlackboardStatus.CONSENSUS,
                "turns": total_turns, "tool_rounds": tool_rounds, "votes": scheduler.vote_state(snapshot),
                "recovery": recovery,
                "intelligibility": "unassessed", "elapsed_s": time.monotonic() - started,
                "usage": {**usage.model_dump(), "total_tokens": usage.total_tokens}}
