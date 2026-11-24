"""FastAPI WebSocket and REST Streaming Server for Real-Time Blackboard Telemetry."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from contracts.schemas import (
    BlackboardSnapshot,
    TelemetryEvent,
    TelemetryEventType,
)
from streaming.manager import telemetry_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("streaming.main")

app = FastAPI(
    title="Intelligible Blackboard Telemetry API",
    description="Real-time WebSocket streaming service broadcasting blackboard mutations, agent queues, and counterfactual events.",
    version="1.0.0",
)

# Enable CORS for the React/Vite development server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EventIngestPayload(BaseModel):
    """Payload format for ingesting telemetry events via HTTP POST."""
    event_type: TelemetryEventType
    session_id: str
    agent_id: Optional[str] = None
    payload: Dict[str, Any] = {}


@app.get("/health", tags=["System"])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint providing status and connection metrics."""
    return {
        "status": "healthy",
        "service": "streaming-telemetry",
        "global_subscribers": len(telemetry_manager.global_connections),
        "total_sessions_tracked": len(telemetry_manager.session_history),
    }


@app.get("/api/sessions", tags=["Sessions"])
async def list_sessions() -> List[Dict[str, Any]]:
    """List all tracked reasoning sessions with event statistics."""
    return await telemetry_manager.list_active_sessions()


@app.get("/api/sessions/{session_id}/events", response_model=List[TelemetryEvent], tags=["Sessions"])
async def get_session_events(session_id: str) -> List[TelemetryEvent]:
    """Retrieve the chronological telemetry event history for a given session."""
    events = await telemetry_manager.get_session_history(session_id)
    if not events and session_id not in telemetry_manager.session_connections:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No telemetry events found for session '{session_id}'.",
        )
    return events


@app.get("/api/sessions/{session_id}/snapshot", response_model=Optional[BlackboardSnapshot], tags=["Sessions"])
async def get_session_snapshot(session_id: str) -> Optional[BlackboardSnapshot]:
    """Retrieve the latest blackboard snapshot cached for a session."""
    snapshot = await telemetry_manager.get_latest_snapshot(session_id)
    if not snapshot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No snapshot cached for session '{session_id}'.",
        )
    return snapshot


@app.get("/api/sessions/{session_id}/graph", tags=["Sessions"])
async def get_session_graph(session_id: str) -> Dict[str, Any]:
    """
    Retrieve React Flow compatible DAG topology (nodes & edges) for a session.
    Extracts contributions, PXP tags, and transitions formatted for interactive rendering.
    """
    snapshot = await telemetry_manager.get_latest_snapshot(session_id)
    history = await telemetry_manager.get_session_history(session_id)

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    contribs: List[Any] = list(snapshot.contributions) if snapshot else []

    if not contribs:
        for evt in history:
            if evt.payload and "contribution" in evt.payload:
                contribs.append(evt.payload["contribution"])

    for idx, c in enumerate(contribs):
        cid = getattr(c, "contribution_id", None) or (c.get("contribution_id") if isinstance(c, dict) else f"node_{idx}")
        agent = getattr(c, "agent_id", None) or (c.get("agent_id") if isinstance(c, dict) else "Agent")
        tag = getattr(c, "tag", None)
        tag_val = tag.value if hasattr(tag, "value") else str(tag or (c.get("tag") if isinstance(c, dict) else "RATIFY"))
        pid = getattr(c, "target_contribution_id", None) or (c.get("target_contribution_id") if isinstance(c, dict) else None)
        is_cf = getattr(c, "is_counterfactual", False) or (c.get("is_counterfactual", False) if isinstance(c, dict) else False)

        claim = ""
        if hasattr(c, "payload") and getattr(c.payload, "prediction", None):
            claim = c.payload.prediction.claim
        elif isinstance(c, dict) and "payload" in c:
            claim = c["payload"].get("prediction", {}).get("claim", "")
        elif hasattr(c, "prediction"):
            claim = str(c.prediction)

        nodes.append({
            "id": str(cid),
            "label": f"{agent}: {tag_val}",
            "position": {"x": idx * 240, "y": 120 if not is_cf else 280},
            "data": {
                "turn_index": idx + 1,
                "agent_id": str(agent),
                "tag": tag_val,
                "claim": claim,
                "is_counterfactual": is_cf,
            },
        })

        if pid:
            edges.append({
                "id": f"e_{pid}_{cid}",
                "source": str(pid),
                "target": str(cid),
                "label": tag_val,
                "animated": is_cf,
            })
        elif idx > 0:
            prev_contrib = contribs[idx - 1]
            prev_id = getattr(prev_contrib, "contribution_id", None) or (
                prev_contrib.get("contribution_id") if isinstance(prev_contrib, dict) else f"node_{idx-1}"
            )
            edges.append({
                "id": f"e_{prev_id}_{cid}",
                "source": str(prev_id),
                "target": str(cid),
                "label": tag_val,
                "animated": is_cf,
            })

    return {
        "session_id": session_id,
        "nodes": nodes,
        "edges": edges,
        "total_nodes": len(nodes),
        "total_edges": len(edges),
    }


@app.post("/api/events", response_model=TelemetryEvent, status_code=status.HTTP_201_CREATED, tags=["Telemetry"])
async def ingest_event(event_data: EventIngestPayload) -> TelemetryEvent:
    """
    Ingest a telemetry event from the scheduler, agent pipeline, or counterfactual sandbox.
    Broadcasts immediately over active WebSockets and buffers in session history.
    """
    event = await telemetry_manager.emit(
        event_type=event_data.event_type,
        session_id=event_data.session_id,
        agent_id=event_data.agent_id,
        payload=event_data.payload,
    )
    return event


@app.post("/api/sessions/{session_id}/snapshot", status_code=status.HTTP_200_OK, tags=["Telemetry"])
async def update_snapshot(session_id: str, snapshot: BlackboardSnapshot) -> Dict[str, str]:
    """Broadcast an explicit blackboard snapshot mutation event."""
    telemetry_manager.latest_snapshots[session_id] = snapshot
    await telemetry_manager.emit(
        event_type=TelemetryEventType.STATE_MUTATION,
        session_id=session_id,
        payload={"snapshot": snapshot.model_dump(mode="json")},
    )
    return {"message": "Snapshot updated and broadcasted."}


@app.websocket("/ws/telemetry")
async def websocket_global_telemetry(websocket: WebSocket) -> None:
    """Global WebSocket stream delivering all telemetry events across all sessions."""
    await telemetry_manager.connect(websocket)
    try:
        active = await telemetry_manager.list_active_sessions()
        await websocket.send_json({
            "type": "CONNECTION_ESTABLISHED",
            "scope": "global",
            "sessions": active,
        })
        while True:
            data = await websocket.receive_text()
            if data.strip().lower() in ("ping", '{"type":"ping"}'):
                await websocket.send_text('{"type":"pong"}')
    except WebSocketDisconnect:
        await telemetry_manager.disconnect(websocket)
    except Exception as exc:
        logger.error("Global WebSocket unexpected error: %s", exc)
        await telemetry_manager.disconnect(websocket)


@app.websocket("/ws/telemetry/{session_id}")
async def websocket_session_telemetry(websocket: WebSocket, session_id: str) -> None:
    """Session-specific WebSocket stream delivering telemetry events for a specific blackboard session."""
    await telemetry_manager.connect(websocket, session_id=session_id)
    try:
        history = await telemetry_manager.get_session_history(session_id)
        snapshot = await telemetry_manager.get_latest_snapshot(session_id)
        await websocket.send_json({
            "type": "SESSION_ATTACHED",
            "session_id": session_id,
            "history_count": len(history),
            "snapshot": snapshot.model_dump(mode="json") if snapshot else None,
            "events": [e.model_dump(mode="json") for e in history],
        })

        while True:
            data = await websocket.receive_text()
            if data.strip().lower() in ("ping", '{"type":"ping"}'):
                await websocket.send_text('{"type":"pong"}')
    except WebSocketDisconnect:
        await telemetry_manager.disconnect(websocket, session_id=session_id)
    except Exception as exc:
        logger.error("Session WebSocket (%s) unexpected error: %s", session_id, exc)
        await telemetry_manager.disconnect(websocket, session_id=session_id)


# ---------------------------------------------------------------------------
# Benchmark run state (in-memory; suitable for single-process deployments)
# ---------------------------------------------------------------------------
_benchmark_results: List[Dict[str, Any]] = []
_benchmark_running: bool = False


class BenchmarkRunRequest(BaseModel):
    """Parameters for triggering an in-process mock ablation batch."""
    trials: int = 8
    densities: List[float] = [0.0, 0.33, 0.66, 1.0]
    dataset: str = "mscore"


@app.post("/api/benchmarks/run", tags=["Benchmarks"])
async def trigger_benchmark_run(
    request: BenchmarkRunRequest,
    background_tasks: BackgroundTasks,
) -> Dict[str, Any]:
    """
    Trigger a mock ablation benchmark batch in the background.
    Uses the BatchTrialRunner simulation engine (no GPU required).
    Results are stored in memory and retrievable via GET /api/benchmarks/results.
    """
    global _benchmark_running
    if _benchmark_running:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A benchmark batch is already running. Poll /api/benchmarks/results for status.",
        )
    _benchmark_running = True

    async def _run() -> None:
        global _benchmark_results, _benchmark_running
        try:
            from benchmarks.mscore import BatchTrialRunner, MSCoReIngestor
            items = MSCoReIngestor.get_sample_dataset()
            runner = BatchTrialRunner(streaming_enabled=True)
            # Restrict to `trials` items to keep latency reasonable
            configs = runner.generate_ablation_matrix(
                items[: max(1, request.trials // 4)],
                densities=request.densities,
            )
            results: List[TrialResult] = await runner.run_batch(configs)
            _benchmark_results = [r.model_dump(mode="json") for r in results]
        except Exception as exc:
            logger.error("Benchmark batch failed: %s", exc)
        finally:
            _benchmark_running = False

    background_tasks.add_task(_run)
    return {
        "status": "started",
        "message": f"Benchmark batch queued ({request.trials} trials, densities={request.densities}). "
                   "Poll GET /api/benchmarks/results for live results.",
    }


@app.get("/api/benchmarks/results", tags=["Benchmarks"])
async def get_benchmark_results() -> Dict[str, Any]:
    """
    Return the latest ablation benchmark results together with aggregated KPI summary.
    Returns an empty list while a run is in-flight (poll until running=false).
    """
    summary: Dict[str, Any] = {}
    if _benchmark_results:
        from analytics.metrics import AblationAnalyzer, summarize_results
        from contracts.schemas import TrialResult
        try:
            parsed = [TrialResult.model_validate(r) for r in _benchmark_results]
            df = summarize_results(parsed)
            analyzer = AblationAnalyzer(df)
            density_summary = analyzer.get_density_summary()
            resolution = analyzer.compute_deadlock_resolution_efficiency()
            # Convert to plain dict for JSON serialisation
            summary = {
                "density_rows": density_summary.to_dict(orient="records"),
                "resolution_efficiency": {str(k): v for k, v in resolution.items()},
                "total_trials": len(_benchmark_results),
                "peak_consensus_pct": float(density_summary["consensus_rate_pct"].max()),
                "peak_ultra_strong_pct": float(density_summary["ultra_strong_pct"].max()),
                "peak_deadlock_recovery_pct": float(
                    density_summary["deadlock_rate_pct"].max()
                ),
            }
        except Exception as exc:
            logger.warning("Could not build summary: %s", exc)

    return {
        "running": _benchmark_running,
        "result_count": len(_benchmark_results),
        "results": _benchmark_results,
        "summary": summary,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("streaming.main:app", host="0.0.0.0", port=8000, reload=True)
