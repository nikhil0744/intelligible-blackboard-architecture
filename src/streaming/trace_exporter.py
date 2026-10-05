"""
src/streaming/trace_exporter.py
Telemetry and Trace Exporter for Intelligible Blackboard Architecture.

Converts live blackboard reasoning sessions and fixture executions into
interactive telemetry traces for visualizer.html and Google Colab presentation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from contracts.schemas import (
    BlackboardSnapshot,
    BlackboardStatus,
    PXPTag,
)


def export_snapshot(
    snap: BlackboardSnapshot,
    output_path: str = "outputs/live_trace.json",
    hardware_info: Optional[str] = None,
    custom_title: Optional[str] = None,
    status_override: Optional[Any] = None,
) -> Dict[str, Any]:
    """Convert a BlackboardSnapshot into a standardized live visualizer trace JSON."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Identify participating agents
    agent_names: List[str] = []
    for c in snap.contributions:
        if c.agent_id not in agent_names:
            agent_names.append(c.agent_id)
    for a in snap.active_agents:
        if a not in agent_names:
            agent_names.append(a)

    # Ensure at least 3 agent slots for UI display
    while len(agent_names) < 3:
        agent_names.append(f"agent_{len(agent_names)}")

    agent_id_to_idx = {name: idx for idx, name in enumerate(agent_names[:3])}

    # Format agent metadata
    def format_agent_role(name: str) -> str:
        lower = name.lower()
        if "cardio" in lower:
            return "CARDIOLOGIST"
        elif "pulmon" in lower:
            return "PULMONOLOGIST"
        elif "attending" in lower or "arbiter" in lower:
            return "ARBITER"
        elif "pharm" in lower or "critic" in lower:
            return "CRITIC"
        elif "clinician" in lower or "expert" in lower or "primary" in lower:
            return "PRIMARY"
        elif "counterfactual" in lower or "cf" in lower:
            return "COUNTERFACTUAL"
        return "SPECIALIST"

    agents_list = [
        {"name": name, "role": format_agent_role(name)}
        for name in agent_names[:3]
    ]

    # 2. Build Step 0 (Init)
    desc = snap.task_description or "No task description"
    task_preview = f"{desc[:87]}..." if len(desc) > 90 else desc

    steps: List[Dict[str, Any]] = [
        {
            "step": 0,
            "turn": "T0 • INIT",
            "narrative": f"Blackboard session initialized. Task: '{task_preview}'",
            "agentActive": None,
            "tag": "IDLE",
            "attribution": "Blackboard Engine (Initialized)",
            "claim": snap.task_description,
            "rationale": "Session opened. Waiting for specialist assertions on the shared blackboard.",
            "status": "ACTIVE",
            "tokens": "0 tok",
            "deadlock": False,
            "sandboxVisible": False,
        }
    ]

    has_deadlock = False
    deadlock_step_idx = -1
    had_sandbox = False
    total_tokens = 0
    recent_tags: List[str] = []

    # 3. Build deliberation steps
    for idx, c in enumerate(snap.contributions):
        turn_num = idx + 1
        agent_idx = agent_id_to_idx.get(c.agent_id, None)
        tag_str = c.tag.value if hasattr(c.tag, "value") else str(c.tag)
        recent_tags.append(tag_str)

        # Check for deadlock condition (2 consecutive REFUTE/REJECT)
        is_step_deadlock = False
        step_deadlock_msg = None
        if len(recent_tags) >= 2 and all(t in ("REFUTE", "REJECT") for t in recent_tags[-2:]):
            has_deadlock = True
            is_step_deadlock = True
            deadlock_step_idx = turn_num
            step_deadlock_msg = f"Circular {tag_str} detected! Disagreement between agents on active claim."

        # Check for counterfactual or sandbox origin
        is_cf = c.is_counterfactual or c.metadata.get("counterfactual_resolution", False)
        if is_cf:
            had_sandbox = True

        status_str = "ACTIVE"
        if is_step_deadlock:
            status_str = "DEADLOCK DETECTED"
        elif is_cf:
            status_str = "DEADLOCK RESOLVED"
        elif tag_str == "RATIFY" and idx == len(snap.contributions) - 1:
            status_str = "CONSENSUS REACHED"

        tok_cost = c.token_usage or 0
        total_tokens += tok_cost
        if c.latency_ms is not None:
            latency_s = (c.latency_ms / 1000.0) if c.latency_ms > 50 else c.latency_ms
            latency_str = f" • {latency_s:.1f}s"
        else:
            latency_str = ""

        narrative = f"{c.agent_id} submitted a {tag_str} assertion to the blackboard."
        if is_cf:
            narrative = f"⚡ Counterfactual resolution from isolated sandbox injected back to blackboard by {c.agent_id}."
        elif is_step_deadlock:
            narrative = f"⚠️ Deadlock detected! {c.agent_id} issued a {tag_str}, resulting in an unresolvable clash."
        elif tag_str == "RATIFY":
            narrative = f"✅ {c.agent_id} agreed with prior claim and issued RATIFY."

        claim_text = c.payload.prediction.claim if c.payload and c.payload.prediction else "No assertion text"
        rationale_text = c.payload.explanation.rationale if c.payload and c.payload.explanation else "No rationale provided"

        steps.append({
            "step": turn_num,
            "turn": f"T{turn_num} • {c.agent_id.upper()}",
            "narrative": narrative,
            "agentActive": agent_idx,
            "tag": tag_str,
            "attribution": f"{c.agent_id} (Turn {c.turn_index})",
            "claim": claim_text,
            "rationale": rationale_text,
            "confidence": getattr(c.payload.prediction, "confidence", 1.0) if c.payload and c.payload.prediction else 1.0,
            "status": status_str,
            "tokens": f"{tok_cost:,} tok{latency_str}",
            "deadlock": is_step_deadlock,
            "deadlockMsg": step_deadlock_msg,
            "sandboxVisible": had_sandbox,
        })

    # 4. Final summary step if consensus or termination reached
    if status_override is not None:
        final_status_val = status_override.value if hasattr(status_override, "value") else str(status_override)
    else:
        final_status_val = snap.status.value if hasattr(snap.status, "value") else str(snap.status)

    if final_status_val in ("CONSENSUS", "RESOLVED", "DEADLOCK") or len(snap.contributions) > 1:
        final_turn_idx = len(steps)
        is_consensus = final_status_val in ("CONSENSUS", "RESOLVED") or any(
            s["tag"] == "RATIFY" for s in steps[-2:]
        )
        
        active_claim_str = snap.active_claim.claim if snap.active_claim else (
            steps[-1]["claim"] if steps else "Final deliberated claim"
        )

        steps.append({
            "step": final_turn_idx,
            "turn": f"T{final_turn_idx} • FINAL SUMMARY",
            "narrative": f"🏁 Deliberation concluded. Status: {final_status_val}. Total tokens consumed: {total_tokens:,}.",
            "agentActive": None,
            "tag": "RATIFY" if is_consensus else "DEADLOCK",
            "attribution": "Blackboard Consensus Engine",
            "claim": f"FINAL RESULT: {active_claim_str}",
            "rationale": f"Total turns: {len(snap.contributions)} | Total tokens: {total_tokens:,} | Final status: {final_status_val}.",
            "status": "CONSENSUS REACHED" if is_consensus else final_status_val,
            "tokens": f"{total_tokens:,} tok (Session Total)",
            "deadlock": not is_consensus and has_deadlock,
            "sandboxVisible": had_sandbox,
        })

    # 5. Sandbox metadata
    if had_sandbox or has_deadlock:
        sandbox_meta = {
            "branch": "ISOLATED COUNTERFACTUAL SANDBOX (Live Session)",
            "subtitle": f"Attributed collision at Turn {max(1, deadlock_step_idx)} • Deep-Copy Cloned DAG",
            "steps": [
                {
                    "title": "[1] Divergence Attribution",
                    "desc": f"Identified Turn {max(1, deadlock_step_idx)} as conflicting state. Cloned blackboard history into isolated sandbox.",
                },
                {
                    "title": "[2] What-If Simulation",
                    "desc": "Simulated alternative counterfactual prompt: Exploring hybrid compromise between competing specialists.",
                },
                {
                    "title": "[3] Peer Ratification & Merge",
                    "desc": "Validated revised claim inside sandbox and merged back to main branch DAG.",
                },
            ],
        }
    else:
        sandbox_meta = {
            "branch": "COUNTERFACTUAL ENGINE (Dormant - No Deadlock)",
            "subtitle": "Direct Unanimous Consensus • Zero Interventions Needed",
            "steps": [
                {
                    "title": "[1] Conflict Monitor",
                    "desc": "Monitored turn-by-turn PXP tags for REFUTE / REJECT cycles. Zero conflicts detected.",
                },
                {
                    "title": "[2] Linear Trajectory",
                    "desc": "All specialists unanimously ratified the proposed claim.",
                },
                {
                    "title": "[3] Final Consensus",
                    "desc": f"Session concluded with direct consensus in {len(snap.contributions)} turns.",
                },
            ],
        }

    trace_data: Dict[str, Any] = {
        "title": custom_title or f"Live Run: {snap.task_id}",
        "task": snap.task_description,
        "sessionId": snap.session_id,
        "hardware": hardware_info or "NVIDIA T4 • Live GPU Inference",
        "agents": agents_list,
        "sandbox": sandbox_meta,
        "steps": steps,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(trace_data, f, indent=2)

    return trace_data


def export_from_board(
    board: Any,
    session_id: str,
    output_path: str = "outputs/live_trace.json",
    hardware_info: Optional[str] = None,
    custom_title: Optional[str] = None,
    status_override: Optional[Any] = None,
) -> Dict[str, Any]:
    """Extract snapshot from an InMemoryBoard and export visualizer live trace."""
    snap = board.get_snapshot(session_id)
    return export_snapshot(
        snap=snap,
        output_path=output_path,
        hardware_info=hardware_info,
        custom_title=custom_title,
        status_override=status_override,
    )


def export_from_spike_blackboard(
    board: Any,
    sandbox: Optional[Any] = None,
    winning_turn: Optional[Any] = None,
    output_path: str = "outputs/live_trace.json",
) -> Dict[str, Any]:
    """Export trace from Student 3's spike_starter InMemoryBlackboard fixture."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    state = board.get_state()
    entries = board.get_entries("main")

    agents_list = [
        {"name": "agent_cardio", "role": "CARDIOLOGIST"},
        {"name": "agent_pulmo", "role": "PULMONOLOGIST"},
        {"name": "agent_attending", "role": "ARBITER"},
    ]

    agent_map = {
        "agent_cardio": 0,
        "agent_pulmo": 1,
        "agent_attending": 2,
    }

    steps: List[Dict[str, Any]] = [
        {
            "step": 0,
            "turn": "T0 • INIT",
            "narrative": "Loaded recorded clinical deadlock scenario into blackboard.",
            "agentActive": None,
            "tag": "IDLE",
            "attribution": "Fixtures (System)",
            "claim": state.problem_statement,
            "rationale": f"Ground truth target: {state.ground_truth}",
            "status": "ACTIVE",
            "tokens": "0 tok",
            "deadlock": False,
            "sandboxVisible": False,
        }
    ]

    for idx, e in enumerate(entries):
        step_num = idx + 1
        agent_idx = agent_map.get(e.agent_id, 0)
        tag_val = e.tag.value if hasattr(e.tag, "value") else str(e.tag)
        is_cf = e.metadata.get("counterfactual_resolution", False)
        is_deadlock = (step_num == 4)  # Step 4 is the clash in sample_deadlocks

        narrative = f"{e.agent_id} asserted: {e.prediction[:70]}..."
        if is_deadlock:
            narrative = "⚠️ Deadlock detected! Pulmonologist rejects Cardiologist's diagnosis."
        elif is_cf:
            narrative = "⚡ Winning revision from isolated sandbox injected back to main blackboard!"

        steps.append({
            "step": step_num,
            "turn": f"T{step_num} • {e.agent_id.upper()}",
            "narrative": narrative,
            "agentActive": agent_idx,
            "tag": tag_val,
            "attribution": f"{e.agent_id} (Step {e.step_number})",
            "claim": e.prediction,
            "rationale": e.explanation,
            "confidence": getattr(e, "confidence", 0.95),
            "status": "DEADLOCK DETECTED" if is_deadlock else ("DEADLOCK RESOLVED" if is_cf else "ACTIVE"),
            "tokens": "1,420 tok (Mock)",
            "deadlock": is_deadlock,
            "deadlockMsg": "Circular REJECT loop (Step 3 ↔ Step 4). Time-travel sandbox required." if is_deadlock else None,
            "sandboxVisible": is_cf or step_num >= 4,
        })

    trace_data: Dict[str, Any] = {
        "title": "Live Run: Spike Deadlock Recovery",
        "task": state.problem_statement,
        "sessionId": state.session_id,
        "hardware": "Fixture Engine • Deterministic Spike Starter",
        "agents": agents_list,
        "sandbox": {
            "branch": "sandbox_cf_spike (Cloned at Step 2)",
            "subtitle": "Attribution: Rolled back to Step 2 • Deep-Copy Cloned DAG",
            "steps": [
                {
                    "title": "[1] Divergence Attribution",
                    "desc": "Identified Step 2 as root clash point. Cloned history into deep-copy sandbox.",
                },
                {
                    "title": "[2] What-If Simulation",
                    "desc": "Simulated hypothetical prompt: 'What if the Cardiologist conceded on IPF and suggested secondary monitoring?'",
                },
                {
                    "title": "[3] Peer Ratification & Injection",
                    "desc": "Synthesized resolution ratified with confidence 0.95 and injected back to live board!",
                },
            ],
        },
        "steps": steps,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(trace_data, f, indent=2)

    return trace_data
