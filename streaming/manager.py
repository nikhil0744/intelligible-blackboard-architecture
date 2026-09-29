"""WebSocket Connection and Telemetry Event Manager for Student 4 streaming service."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Set
from fastapi import WebSocket

from contracts.schemas import (
    BlackboardSnapshot,
    TelemetryEvent,
    TelemetryEventType,
)

logger = logging.getLogger("streaming.manager")


class ConnectionManager:
    """
    Manages active WebSocket connections, session rooms, and event history buffers.
    Incorporates per-socket write locks and bounded circular deques to prevent memory leaks
    and concurrent write collisions in Starlette WebSockets.
    """

    def __init__(self, history_limit: int = 1000, max_tracked_sessions: int = 500):
        # Global connections receive all telemetry events
        self.global_connections: Set[WebSocket] = set()
        # Session-scoped connections: session_id -> Set[WebSocket]
        self.session_connections: Dict[str, Set[WebSocket]] = defaultdict(set)
        # Event history buffer with bounded deque: session_id -> Deque[TelemetryEvent]
        self.session_history: Dict[str, Deque[TelemetryEvent]] = defaultdict(
            lambda: deque(maxlen=history_limit)
        )
        # Latest snapshot cache per session
        self.latest_snapshots: Dict[str, BlackboardSnapshot] = {}
        # Concurrency write lock per WebSocket connection
        self._socket_locks: Dict[WebSocket, asyncio.Lock] = {}
        # Global state lock
        self._lock = asyncio.Lock()
        self.history_limit = history_limit
        self.max_tracked_sessions = max_tracked_sessions

    def _get_socket_lock(self, websocket: WebSocket) -> asyncio.Lock:
        if websocket not in self._socket_locks:
            self._socket_locks[websocket] = asyncio.Lock()
        return self._socket_locks[websocket]

    async def connect(self, websocket: WebSocket, session_id: Optional[str] = None) -> None:
        """Accept WebSocket and register under global or session-specific group."""
        await websocket.accept()
        async with self._lock:
            self._socket_locks[websocket] = asyncio.Lock()
            if session_id:
                self.session_connections[session_id].add(websocket)
                logger.info(
                    "WebSocket connected for session %s (total: %d)",
                    session_id,
                    len(self.session_connections[session_id]),
                )
            else:
                self.global_connections.add(websocket)
                logger.info("WebSocket connected globally (total: %d)", len(self.global_connections))

    async def disconnect(self, websocket: WebSocket, session_id: Optional[str] = None) -> None:
        """Remove WebSocket connection on disconnect or network error."""
        async with self._lock:
            if session_id and session_id in self.session_connections:
                self.session_connections[session_id].discard(websocket)
                if not self.session_connections[session_id]:
                    del self.session_connections[session_id]
                logger.info("WebSocket removed from session %s", session_id)
            self.global_connections.discard(websocket)
            self._socket_locks.pop(websocket, None)

    async def get_session_history(self, session_id: str) -> List[TelemetryEvent]:
        """Retrieve recorded telemetry events for a given session thread-safely."""
        async with self._lock:
            return list(self.session_history.get(session_id, []))

    async def get_latest_snapshot(self, session_id: str) -> Optional[BlackboardSnapshot]:
        """Retrieve the most recent cached snapshot for a session thread-safely."""
        async with self._lock:
            return self.latest_snapshots.get(session_id)

    async def list_active_sessions(self) -> List[Dict[str, Any]]:
        """List summary of known sessions with event counts and active subscribers under lock."""
        async with self._lock:
            known_sessions = set(self.session_history.keys()) | set(self.session_connections.keys())
            summaries = []
            for sid in sorted(known_sessions):
                history = list(self.session_history.get(sid, []))
                last_event = history[-1] if history else None
                snapshot = self.latest_snapshots.get(sid)
                summaries.append({
                    "session_id": sid,
                    "event_count": len(history),
                    "active_subscribers": len(self.session_connections.get(sid, set())),
                    "last_event_type": last_event.event_type.value if last_event else None,
                    "last_timestamp": last_event.timestamp.isoformat() if last_event else None,
                    "status": snapshot.status.value if snapshot else "UNKNOWN",
                })
            return summaries

    async def broadcast_event(self, event: TelemetryEvent) -> None:
        """
        Buffer the event and broadcast it to relevant WebSocket clients.
        Serializes into standard JSON format conforming to Pydantic TelemetryEvent.
        Uses per-socket locks to eliminate concurrent send_text() collisions.
        """
        session_id = event.session_id

        # Buffer in bounded history and evict oldest if too many sessions
        async with self._lock:
            self.session_history[session_id].append(event)

            if len(self.session_history) > self.max_tracked_sessions:
                # Evict oldest session that has no active subscribers
                for sid in list(self.session_history.keys()):
                    if sid not in self.session_connections and sid != session_id:
                        del self.session_history[sid]
                        self.latest_snapshots.pop(sid, None)
                        break

            # If payload contains a snapshot, cache it
            if "snapshot" in event.payload and isinstance(event.payload["snapshot"], dict):
                try:
                    self.latest_snapshots[session_id] = BlackboardSnapshot.model_validate(
                        event.payload["snapshot"]
                    )
                except Exception as exc:
                    logger.debug("Could not parse snapshot from payload: %s", exc)

            # Identify target websockets: global subscribers + session-scoped subscribers
            targets = set(self.global_connections)
            if session_id in self.session_connections:
                targets.update(self.session_connections[session_id])

        if not targets:
            return

        message_text = event.model_dump_json()
        dead_connections: List[tuple[WebSocket, Optional[str]]] = []

        async def _safe_send(ws: WebSocket):
            lock = self._get_socket_lock(ws)
            async with lock:
                try:
                    await ws.send_text(message_text)
                except Exception as exc:
                    logger.warning("Failed to send WebSocket message: %s", exc)
                    dead_connections.append((ws, session_id))

        await asyncio.gather(*[_safe_send(ws) for ws in targets], return_exceptions=True)

        for ws, sid in dead_connections:
            await self.disconnect(ws, sid)

    async def emit(
        self,
        event_type: TelemetryEventType,
        session_id: str,
        agent_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> TelemetryEvent:
        """Helper to create, record, and broadcast a TelemetryEvent with agent attribution."""
        event = TelemetryEvent(
            event_type=event_type,
            session_id=session_id,
            agent_id=agent_id,
            timestamp=datetime.now(timezone.utc),
            payload=payload or {},
        )
        await self.broadcast_event(event)
        return event

    def clear(self) -> None:
        """Reset all in-memory buffers (primarily for testing)."""
        self.session_history.clear()
        self.latest_snapshots.clear()
        self.global_connections.clear()
        self.session_connections.clear()
        self._socket_locks.clear()


# Global singleton instance for the service
telemetry_manager = ConnectionManager()
