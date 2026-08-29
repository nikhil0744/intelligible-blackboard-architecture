# Intelligible Blackboard Architecture with Counterfactual Agents

This repository implements an **Intelligible Blackboard Architecture** for multi-agent systems, integrating:
- **Two-Way Intelligibility Protocol (PXP)** (Baskar et al.) pairing predictions ("what") with explanations ("why").
- **Retrospective Counterfactual Reasoning** for deadlock resolution and credit assignment.
- **Deterministic Scheduling** and concurrency control.
- **Real-Time Visualization Matrix** with dynamic node graphs and split-screen alternative timeline exploration.

## Project Structure
- `blackboard/`: Core state store, PXP protocol validator, and deterministic scheduler.
- `agents/`: PEX agent classes, persona matrix, and local LLM brokers (Ollama / LiteLLM).
- `counterfactual/`: Retrospective sandbox and history rollback simulation module.
- `frontend/`: Real-time web visualization dashboard (React + React Flow / WebSocket).
- `benchmarks/`: Data ingestors and automated ablation test suites (KramaBench, MSCoRe, MedAgentBench).
