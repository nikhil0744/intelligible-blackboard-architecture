"""Read verified records and export recorded playback without the legacy trace synthesizer."""
from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

from .actions import benchmark_claim_key
from .persistence import RunStore, read_result, write_json


def read_trace(run_dir, attempt_id=None):
    store = RunStore(run_dir)
    if attempt_id is None:
        committed = [r for r in store.attempts() if r.get("committed", True)]
        if not committed:
            raise ValueError("This run has no committed attempt to inspect")
        attempt_id = committed[-1]["attempt_id"]
    if Path(attempt_id).name != attempt_id:
        raise ValueError("invalid attempt ID")
    path = store.root / "attempts" / attempt_id
    result = read_result(path)
    events = [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]
    return store, result, events


def trace_table(run_dir, attempt_id=None):
    """Notebook view: public contributions and tool activity, in recorded order."""
    _, _, events = read_trace(run_dir, attempt_id)
    rows = []
    for event in events:
        kind = event["kind"]
        if kind == "contribution":
            c = event["contribution"]
            rows.append({"time": event["time"], "kind": kind, "agent": c["agent_id"], "tag/tool": c["tag"],
                         "id": c["contribution_id"], "target": c["target_contribution_id"],
                         "content": c["payload"]["prediction"]["claim"],
                         "explanation/observation": c["payload"]["explanation"]["rationale"]})
        elif kind in {"tool_requested", "tool_observed", "tool_request_blocked", "tools_closed", "evidence_ready",
                      "query_scope_mismatch", "discussion_stalled", "session_stopped", "evaluation"} or kind.startswith("recovery_"):
            rows.append({"time": event["time"], "kind": kind, "agent": event.get("agent_id"),
                         "tag/tool": event.get("name"), "id": event.get("request_id"), "target": None,
                         "content": json.dumps({k: v for k, v in event.items() if k not in {"kind", "time"}}, ensure_ascii=False),
                         "explanation/observation": ""})
    return rows


def _tool_request_detail(arguments):
    """Render the selected CSV operation without exposing JSON on the default view."""
    lines = []
    if arguments.get("file"):
        lines.append(f"Input: {arguments['file']}")
    if arguments.get("header") is not None:
        lines.append(f"CSV header row: {arguments['header']} (starting at 0)")
    operator_phrases = {"eq": "equals", "ne": "does not equal", "in": "is one of",
                        "contains": "contains", "gt": "is greater than", "gte": "is at least",
                        "ge": "is at least", "lt": "is less than", "lte": "is at most",
                        "le": "is at most"}
    for condition in arguments.get("filters") or []:
        if not isinstance(condition, dict):
            continue
        operator = operator_phrases.get(condition.get("op"), condition.get("op", "?"))
        value = condition.get("value")
        if isinstance(value, list):
            value = ", ".join(map(str, value))
        lines.append(f"Keep rows where {condition.get('column', '?')} {operator} {value}")
    if isinstance(aggregate := arguments.get("aggregate"), dict):
        lines.append(f"Compute {aggregate.get('op', 'an aggregation')} of {aggregate.get('column') or 'matching rows'}")
    if arguments.get("group_by"):
        lines.append("Group by " + ", ".join(map(str, arguments["group_by"])))
    if isinstance(arguments.get("select"), list):
        lines.append("Return " + ", ".join(map(str, arguments["select"])))
    return "\n".join(lines)


def _tool_error(error):
    detail = error.removeprefix("ValueError: ")
    if detail.startswith("Unknown CSV column "):
        column, _, rest = detail.removeprefix("Unknown CSV column ").partition(" at header=")
        header, _, rest = rest.partition(". Parsed columns: ")
        parsed, _, _ = rest.partition(". Revise")
        return ("Column " + column + " was unavailable in the parsed CSV.\n"
                f"Header row: {header}.\nColumns found: {parsed.strip('.')}\n"
                "Inspect the CSV header or choose a column from the parsed list.")
    if error.startswith("KeyError: "):
        return ("The tool could not find column " + error.removeprefix("KeyError: ")
                + ". Inspect the CSV header and use a listed column name.")
    if detail == "tool_timeout":
        return "The CSV tool did not finish within its time limit."
    return detail


def _recorded_timeline(events):
    """Project the event log into readable steps with branch-local claim and vote state."""
    timeline = []
    states = {}
    request_names = {}
    branch = "main"
    for event in events:
        kind = event["kind"]
        branch = event.get("branch") or branch
        state = states.setdefault(branch, {"claim": None, "ratifiers": set(), "linked_ids": set()})
        request_id = event.get("request_id")
        tool_name = event.get("name") or request_names.get(request_id)
        step = {"kind": kind, "time": event["time"], "branch": branch,
                "title": kind.replace("_", " ").capitalize(), "agent": event.get("agent_id"),
                "detail": event.get("reason", ""), "raw": event}

        if kind == "contribution":
            contribution = event["contribution"]
            prediction, explanation = (contribution["payload"][key] for key in ("prediction", "explanation"))
            cid = contribution["contribution_id"]
            target = contribution.get("target_contribution_id")
            tag = contribution["tag"]
            submitted = contribution.get("metadata", {}).get("final_submission", False)
            if tag == "REVISE":
                state.update(claim=prediction["claim"], ratifiers=set(), linked_ids={cid})
            elif tag in {"REFUTE", "REJECT"}:
                state["ratifiers"].clear()
            elif (tag == "RATIFY" and target in state["linked_ids"]
                  and benchmark_claim_key(prediction["claim"]) == benchmark_claim_key(state["claim"])):
                state["ratifiers"].add(contribution["agent_id"])
                state["linked_ids"].add(cid)
            step.update(title="Answer submitted" if submitted else tag.capitalize(),
                        agent=contribution["agent_id"], tag=tag, claim=prediction["claim"],
                        rationale=explanation["rationale"], evidence=explanation.get("evidence", []),
                        contribution_id=cid, target=target)
        elif kind == "tool_requested":
            request_names[request_id] = tool_name
            step.update(title=f"Run {tool_name}", detail=_tool_request_detail(event.get("arguments", {})))
        elif kind == "tool_observed":
            outcome = event.get("result", {})
            data = outcome.get("data")
            step["title"] = f"{(tool_name or 'Tool').capitalize()} result"
            if not outcome.get("ok"):
                step["title"] = f"{(tool_name or 'Tool').capitalize()} failed"
                step["detail"] = _tool_error(outcome.get("error") or "Tool call did not complete.")
            elif isinstance(data, dict) and "value" in data:
                step["detail"] = f"Value: {data['value']} · matched rows: {data.get('matched_rows', 'unknown')}"
            elif isinstance(data, dict) and "columns" in data:
                step["detail"] = ", ".join(str(c) for c in data["columns"])
            elif isinstance(data, dict) and "matched_rows" in data:
                step["detail"] = f"Matched rows: {data['matched_rows']}"
            elif isinstance(data, list):
                step["detail"] = f"{len(data)} input files found"
            elif data is not None:
                step["detail"] = str(data)
        elif kind == "query_scope_mismatch":
            step.update(title="Query scope needs refinement",
                        detail="\n".join(event.get("issues", [])) or "The saved query did not cover the requested data.")
        elif kind == "evidence_ready":
            step.update(title="CSV evidence accepted",
                        detail="The analyst can use this query result to submit an answer.")
        elif kind == "recovery_started":
            step.update(title="Recovery started", claim=event.get("source_claim"),
                        detail="Reopening from checkpoint: " + event.get("checkpoint", "").replace("_", " "))
        elif kind == "recovery_candidate":
            step.update(title="Recovery candidate", claim=event.get("claim"),
                        contribution_id=event.get("contribution_id"))
        elif kind == "recovery_finished":
            step.update(title="Recovery finished", detail=event.get("status", ""))
        elif kind == "session_stopped":
            step.update(title="Run stopped", detail=event.get("status", ""))
        elif kind == "evaluation":
            grade = event["grade"]
            detail = f"Score: {grade['score']}" if grade.get("status") == "scored" else grade.get("status", "")
            step.update(title="Independent grade", detail=detail)
        elif kind == "model_response":
            step["title"] = "Model repair response" if event.get("is_repair") else "Model response"
        step["active_claim"] = state["claim"]
        votes = event.get("votes") or {}
        step["ratifiers"] = votes.get("ratified_agent_ids", sorted(state["ratifiers"]))
        timeline.append(step)
    return timeline


def export_viewer(run_dir, template=None, attempt_id=None):
    store, result, events = read_trace(run_dir, attempt_id)
    first = next(e for e in events if e["kind"] == "session_started")
    agents = [{"name": a["agent_id"], "role": a["role"]} for a in first["roster"]]
    index = {a["name"]: i for i, a in enumerate(agents)}
    readiness = store.manifest["observations"].get("readiness", {})
    gpus = readiness.get("hardware", {}).get("gpus", [])
    hardware = ", ".join(g["name"] for g in gpus) or "GPU not recorded / CPU smoke"
    tokens = result["usage"]["total_tokens"]
    token_text = f"Attempt total: {tokens if tokens is not None else 'unknown'} tokens • {result['elapsed_s']:.2f}s"
    steps = []
    for event in events:
        kind = event["kind"]
        if kind not in {"session_started", "contribution", "tool_requested", "tool_observed", "session_stopped", "evaluation"}:
            continue
        step = {"step": len(steps), "turn": f"Event {len(steps)} • {kind}", "timestamp": event["time"],
                "narrative": kind.replace("_", " "), "agentActive": index.get(event.get("agent_id")),
                "tag": "IDLE", "attribution": event.get("agent_id", "Controller"),
                "claim": "", "rationale": "", "status": "ACTIVE", "tokens": token_text,
                "deadlock": False, "sandboxVisible": False, "thoughtFlows": []}
        if kind == "session_started":
            step["claim"] = first["task"]["question"]
            step["rationale"] = "Recorded baseline session; recovery disabled."
        elif kind == "contribution":
            c = event["contribution"]
            p, x = c["payload"]["prediction"], c["payload"]["explanation"]
            step.update(agentActive=index.get(c["agent_id"]), tag=c["tag"], attribution=c["agent_id"],
                        claim=p["claim"], rationale=x["rationale"], contribution_id=c["contribution_id"],
                        target_contribution_id=c["target_contribution_id"])
            step["narrative"] = f"{c['agent_id']} • {c['tag']} • target {c['target_contribution_id'] or 'none'}"
            step["thoughtFlows"] = [None] * len(agents)
            step["thoughtFlows"][index[c["agent_id"]]] = {
                "evidence": "; ".join(x["evidence"]) or "No evidence cited",
                "hypothesis": p["claim"], "deliberation": x["rationale"], "action": c["tag"],
                "conf": str(p["confidence"])}
        elif kind.startswith("tool_"):
            step["claim"] = event.get("name", "Tool observation")
            step["rationale"] = json.dumps(event.get("arguments", event.get("result")), ensure_ascii=False)
            step["narrative"] = f"{kind} • request {event['request_id']}"
        elif kind == "session_stopped":
            step["status"] = event["status"].upper()
            step["deadlock"] = event["status"] == "deadlock"
            step["deadlockMsg"] = "Recorded disagreement; baseline recovery is disabled."
            step["claim"] = f"Final output: {result['output']}"
            step["rationale"] = f"Consensus recorded: {event['consensus']}. Independent grading follows."
        else:
            step["status"] = result["status"].upper()
            step["claim"] = f"Grade: {json.dumps(event['grade'])}"
            step["rationale"] = "Selected profile metric only; consensus and correctness are separate."
        steps.append(step)
    trace = {"mode": "recorded_benchmark", "title": f"{result['task_id']} • {result['attempt_id'][:8]}",
             "task": first["task"]["question"], "sessionId": result["session_id"], "hardware": hardware,
             "agents": agents, "sandbox": None, "steps": steps, "result": result,
             "timeline": _recorded_timeline(events),
             "source": str(store.root), "official_aggregate": False}
    output = store.root / "presentation" / result["attempt_id"]
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "trace.json", trace)
    # Escape '<' so model output cannot close the script element. The viewer renders
    # claims and observations as textContent; no untrusted HTML is generated here.
    encoded = json.dumps(trace, ensure_ascii=True).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    html = (files("benchmarks").joinpath("benchmark_viewer.html").read_text(encoding="utf-8")
            if template is None else Path(template).read_text(encoding="utf-8"))
    if "<!-- BENCHMARK_TRACE_DATA -->" in html:
        html = html.replace("<!-- BENCHMARK_TRACE_DATA -->",
                            '<script id="recorded-data" type="application/json">' + encoded + "</script>", 1)
    elif "window.LIVE_TRACE_DATA" in html and "recorded_benchmark" in html:
        html = html.replace("<head>", "<head>\n<script>window.LIVE_TRACE_DATA=" + encoded + ";</script>", 1)
    else:
        raise ValueError("The supplied template does not support recorded benchmark traces")
    (output / "viewer.html").write_text(html)
    return output / "viewer.html"


def plot_summary(run_dir):
    """Optional static plot derived from verified records, with an explicit denominator."""
    import matplotlib.pyplot as plt
    summary = RunStore(run_dir).summarize()
    labels = ["Correct", "Incorrect", "Ungraded"]
    fig, axis = plt.subplots(figsize=(6, 3))
    axis.bar(labels, [summary["correct"], summary["incorrect"], summary["ungraded"]])
    axis.set(ylabel="Selected tasks", title=f"Pilot profile • {summary['selected']} selected tasks (not an official aggregate)")
    from matplotlib.ticker import MaxNLocator
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    path = Path(run_dir) / "figures/outcomes.png"
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
