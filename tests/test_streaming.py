"""Unit and integration tests for Student 4 WebSocket streaming service and manager."""

import pytest
from fastapi.testclient import TestClient

from contracts.schemas import (
    BlackboardSnapshot,
    BlackboardStatus,
    TelemetryEvent,
    TelemetryEventType,
)
from streaming.main import app
from streaming.manager import ConnectionManager, telemetry_manager


@pytest.fixture(autouse=True)
def reset_telemetry_state():
    """Reset telemetry manager state between tests."""
    telemetry_manager.clear()
    yield
    telemetry_manager.clear()


@pytest.mark.asyncio
async def test_connection_manager_buffering():
    manager = ConnectionManager(history_limit=5)
    sid = "test_session_01"

    # Emit 3 events
    for i in range(3):
        await manager.emit(
            event_type=TelemetryEventType.TURN_DISPATCHED,
            session_id=sid,
            payload={"turn": i + 1},
        )

    history = manager.get_session_history(sid)
    assert len(history) == 3
    assert history[0].payload["turn"] == 1
    assert history[2].payload["turn"] == 3

    # Ensure active session summaries report correctly
    sessions = manager.list_active_sessions()
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == sid
    assert sessions[0]["event_count"] == 3


@pytest.mark.asyncio
async def test_connection_manager_history_limit():
    manager = ConnectionManager(history_limit=3)
    sid = "overflow_session"

    for i in range(5):
        await manager.emit(
            event_type=TelemetryEventType.STATE_MUTATION,
            session_id=sid,
            payload={"step": i},
        )

    history = manager.get_session_history(sid)
    assert len(history) == 3
    # First two should have been evicted
    assert history[0].payload["step"] == 2
    assert history[2].payload["step"] == 4


def test_rest_health_endpoint():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["service"] == "streaming-telemetry"


def test_rest_event_ingest_and_retrieval():
    client = TestClient(app)
    sid = "rest_test_session"

    # Post an event
    post_resp = client.post(
        "/api/events",
        json={
            "event_type": "DEADLOCK_DETECTED",
            "session_id": sid,
            "payload": {"reason": "Test circular refutation"},
        },
    )
    assert post_resp.status_code == 201
    created_event = post_resp.json()
    assert created_event["event_type"] == "DEADLOCK_DETECTED"
    assert created_event["session_id"] == sid

    # Query events for this session
    get_resp = client.get(f"/api/sessions/{sid}/events")
    assert get_resp.status_code == 200
    events = get_resp.json()
    assert len(events) == 1
    assert events[0]["payload"]["reason"] == "Test circular refutation"


def test_websocket_telemetry_flow():
    client = TestClient(app)
    sid = "ws_test_session"

    with client.websocket_connect(f"/ws/telemetry/{sid}") as websocket:
        # Initial frame on connection
        init_frame = websocket.receive_json()
        assert init_frame["type"] == "SESSION_ATTACHED"
        assert init_frame["session_id"] == sid

        # Ingest an event via REST
        client.post(
            "/api/events",
            json={
                "event_type": "CONSENSUS_REACHED",
                "session_id": sid,
                "payload": {"consensus_turns": 4},
            },
        )

        # Receive streamed event on the websocket
        live_frame = websocket.receive_json()
        assert live_frame["event_type"] == "CONSENSUS_REACHED"
        assert live_frame["payload"]["consensus_turns"] == 4
