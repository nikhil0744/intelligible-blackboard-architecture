> Historical contributor guide. Run setup commands from the repository root; see the [current README](../../README.md) for the source layout and setup.

# Student 4: Real-Time Visualization UI & Benchmarking

This module contains the deliverables for **Student 4** in the Intelligible Blackboard Architecture team project.

---

## 1. Architecture & Deliverables

### A. FastAPI WebSocket Streaming Service (`src/streaming/`)
- **Connection Manager (`src/streaming/manager.py`)**: Thread-safe WebSocket connection pool supporting global and session-specific channels, in-memory event buffering, and automatic reconnection synchronization.
- **FastAPI Server (`src/streaming/main.py`)**:
  - `GET /health`: Health and connection telemetry.
  - `GET /api/sessions`: List active reasoning sessions and event statistics.
  - `GET /api/sessions/{session_id}/events`: Session event back-history.
  - `POST /api/events`: Ingest events from upstream Scheduler, Agent Engine, or Counterfactual Sandbox.
  - `WS /ws/telemetry`: Global real-time stream broadcasting all blackboard turns.
  - `WS /ws/telemetry/{session_id}`: Targeted stream for a single session.

### B. Interactive React/Vite Dashboard (`frontend/`)
- **Reasoning Node Graph (`GraphViewer.jsx`)**: Real-time visualization of agent assertions, confidence gauges, and PXP interaction tags:
  - `RATIFY` (Green)
  - `REVISE` (Blue)
  - `REFUTE` (Amber)
  - `REJECT` (Red)
- **Split-Panel View (`SplitPanelView.jsx`)**: Side-by-side comparison of the live blackboard run alongside Retrospective Counterfactual sandbox rollbacks.
- **Bottleneck Monitor (`BottleneckMonitor.jsx`)**: Instant impasse alert banner, contradiction volume meter, and lock state monitor.
- **Playback Controls (`PlaybackControls.jsx`)**: Turn scrubber, play/pause, step forward/back, and 0.5x–5x playback speed.
- **Ablation Dashboard (`AblationDashboard.jsx`)**: Multi-density KPI cards and comparative table across 0%, 33%, 66%, and 100% counterfactual densities.

### C. MSCoRe Ingestion & Shared Batch Trial Runner (`src/benchmarks/mscore.py`)
- **`MSCoReIngestor`**: Parses JSON/JSONL benchmark files and provides built-in clinical/technical test scenarios.
- **`BatchTrialRunner`**: Shared trial execution engine used across all three datasets (MSCoRe, KramaBench, MedAgentBench), modeling counterfactual resolution dynamics and outputting `TrialResult` records.

### D. Analytics & Publication Graphics (`src/analytics/`)
- **`AblationAnalyzer`**: Statistical KPI aggregator (consensus rate, turns, token cost, deadlock recovery rate, Ultra-Strong intelligibility rate).
- **`plotting.py`**: Automated Matplotlib script generating research figures:
  - `fig1_consensus_vs_density.png`
  - `fig2_cost_vs_recovery.png`
  - `fig3_intelligibility_progression.png`

---

## 2. Quick Start Guide

### Step 1: Install Python Dependencies
```bash
pip install -e ".[dev]"
```

### Step 2: Run Tests
```bash
pytest
```
*All 20 unit and integration tests validate streaming, MSCoRe ingestion, and analytics.*

### Step 3: Start the Streaming Backend
```bash
uvicorn streaming.main:app --reload --port 8000
```
Open interactive Swagger API docs at: `http://localhost:8000/docs`

### Step 4: Start the Frontend UI
```bash
cd frontend
npm install
npm run dev
```
Open the interactive dashboard at: `http://localhost:5173`
*(Note: A built-in demo deliberation is pre-configured so you can explore the node graph and split-panel view even before live agents connect).*
