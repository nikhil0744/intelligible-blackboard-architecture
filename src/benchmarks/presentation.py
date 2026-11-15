"""Read verified records and export recorded playback without the legacy trace synthesizer."""
from __future__ import annotations

import json
from pathlib import Path

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
        elif kind in {"tool_requested", "tool_observed", "session_stopped", "evaluation"}:
            rows.append({"time": event["time"], "kind": kind, "agent": event.get("agent_id"),
                         "tag/tool": event.get("name"), "id": event.get("request_id"), "target": None,
                         "content": json.dumps({k: v for k, v in event.items() if k not in {"kind", "time"}}, ensure_ascii=False),
                         "explanation/observation": ""})
    return rows


def export_viewer(run_dir, template, attempt_id=None):
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
             "source": str(store.root), "official_aggregate": False}
    output = store.root / "presentation" / result["attempt_id"]
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "trace.json", trace)
    # Escape '<' so model output cannot close the script element. The viewer renders
    # claims and observations as textContent; no untrusted HTML is generated here.
    encoded = json.dumps(trace, ensure_ascii=True).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    html = Path(template).read_text()
    if "window.LIVE_TRACE_DATA" not in html or "recorded_benchmark" not in html:
        raise ValueError("Use this revision's visualizer.html, which supports recorded benchmark traces")
    html = html.replace("<head>", "<head>\n<script>window.LIVE_TRACE_DATA=" + encoded + ";</script>", 1)
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
