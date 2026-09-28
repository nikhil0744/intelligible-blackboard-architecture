import { useState, useEffect, useRef, useCallback } from 'react';

export function useWebSocket(url, sessionId = null) {
  const [status, setStatus] = useState('DISCONNECTED'); // CONNECTING, CONNECTED, DISCONNECTED, ERROR
  const [events, setEvents] = useState([]);
  const [latestSnapshot, setLatestSnapshot] = useState(null);
  const [activeSessions, setActiveSessions] = useState([]);
  const [deadlockState, setDeadlockState] = useState(null);
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);

  const targetUrl = sessionId
    ? `${url.replace(/\/$/, '')}/${sessionId}`
    : url;

  const connect = useCallback(() => {
    setStatus('CONNECTING');
    try {
      const ws = new WebSocket(targetUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setStatus('CONNECTED');
        console.log(`[WebSocket] Connected to ${targetUrl}`);
      };

      ws.onmessage = (eventMsg) => {
        try {
          const parsed = JSON.parse(eventMsg.data);

          // Handle system control messages
          if (parsed.type === 'CONNECTION_ESTABLISHED') {
            setActiveSessions(parsed.sessions || []);
            return;
          }
          if (parsed.type === 'SESSION_ATTACHED') {
            if (parsed.snapshot) setLatestSnapshot(parsed.snapshot);
            if (parsed.events) setEvents(parsed.events);
            return;
          }

          // Handle TelemetryEvent messages
          if (parsed.event_type) {
            setEvents((prev) => [...prev, parsed]);

            if (parsed.payload?.snapshot) {
              setLatestSnapshot(parsed.payload.snapshot);
            }

            if (parsed.event_type === 'DEADLOCK_DETECTED') {
              setDeadlockState({
                detected: true,
                reason: parsed.payload?.reason || 'Circular REFUTE/REJECT loop detected',
                turn: parsed.payload?.turn,
                timestamp: parsed.timestamp,
              });
            } else if (parsed.event_type === 'COUNTERFACTUAL_RESOLVED' || parsed.event_type === 'CONSENSUS_REACHED') {
              setDeadlockState(null);
            }
          }
        } catch (err) {
          console.warn('[WebSocket] Error parsing frame:', err);
        }
      };

      ws.onerror = (err) => {
        console.error('[WebSocket] Error:', err);
        setStatus('ERROR');
      };

      ws.onclose = () => {
        setStatus('DISCONNECTED');
        console.log('[WebSocket] Connection closed. Scheduling reconnect in 3s...');
        reconnectTimeoutRef.current = setTimeout(() => {
          connect();
        }, 3000);
      };
    } catch (err) {
      console.error('[WebSocket] Instantiation failed:', err);
      setStatus('ERROR');
    }
  }, [targetUrl]);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [connect]);

  const send = useCallback((data) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(typeof data === 'string' ? data : JSON.stringify(data));
    }
  }, []);

  const clearEvents = useCallback(() => {
    setEvents([]);
  }, []);

  return {
    status,
    events,
    latestSnapshot,
    activeSessions,
    deadlockState,
    send,
    clearEvents,
  };
}
