"""FastAPI WebSocket and REST Streaming Server for Real-Time Blackboard Telemetry."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
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
    return telemetry_manager.list_active_sessions()


@app.get("/api/sessions/{session_id}/events", response_model=List[TelemetryEvent], tags=["Sessions"])
async def get_session_events(session_id: str) -> List[TelemetryEvent]:
    """Retrieve the chronological telemetry event history for a given session."""
    events = telemetry_manager.get_session_history(session_id)
    if not events and session_id not in telemetry_manager.session_connections:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No telemetry events found for session '{session_id}'.",
        )
    return events


@app.get("/api/sessions/{session_id}/snapshot", response_model=Optional[BlackboardSnapshot], tags=["Sessions"])
async def get_session_snapshot(session_id: str) -> Optional[BlackboardSnapshot]:
    """Retrieve the latest blackboard snapshot cached for a session."""
    snapshot = telemetry_manager.get_latest_snapshot(session_id)
    if not snapshot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No snapshot cached for session '{session_id}'.",
        )
    return snapshot


@app.post("/api/events", response_model=TelemetryEvent, status_code=status.HTTP_201_CREATED, tags=["Telemetry"])
async def ingest_event(event_data: EventIngestPayload) -> TelemetryEvent:
    """
    Ingest a telemetry event from the scheduler, agent pipeline, or counterfactual sandbox.
    Broadcasts immediately over active WebSockets and buffers in session history.
    """
    event = await telemetry_manager.emit(
        event_type=event_data.event_type,
        session_id=event_data.session_id,
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
        # Send initial welcome and active sessions
        await websocket.send_json({
            "type": "CONNECTION_ESTABLISHED",
            "scope": "global",
            "sessions": telemetry_manager.list_active_sessions(),
        })
        while True:
            # Keep connection open, handle client control commands (e.g. ping)
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
        # On connection, flush existing history so client catches up
        history = telemetry_manager.get_session_history(session_id)
        snapshot = telemetry_manager.get_latest_snapshot(session_id)
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("streaming.main:app", host="0.0.0.0", port=8000, reload=True)
