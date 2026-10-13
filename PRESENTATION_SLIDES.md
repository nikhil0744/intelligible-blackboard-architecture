# Intelligible Multi-Agent Blackboard Architecture
## Mid-Semester Implementation Checkpoint & Proof-of-Work Verification

---

# Slide 1: Title & Project Status

### Intelligible Multi-Agent Blackboard Architecture
**Sub-title:** Mid-Semester Engineering Checkpoint: System Implementation & Proof-of-Work Verification

**Presented by:**
- Amithav C (Blackboard Engine & Concurrency)
- Shiva P (Agent Personas & Dataset Ingestion)
- Nikhil Palakollu (Counterfactual Engine & Sandbox Simulator)
- Harish K (Interactive Visualizer UI & Evaluation)

**Implementation Status:** Core Architecture Deployed • 179 Unit Tests Passing • Real Local Models Integrated (Ollama / Qwen2.5-7B)

> **Presenter Note:** "Good morning everyone. Today we are presenting our mid-semester implementation checkpoint for the Intelligible Multi-Agent Blackboard Architecture. Rather than just presenting concepts, we are excited to show our working software architecture, real local model execution telemetry, and our automated verification test suite."

---

# Slide 2: What We Have Built — System Architecture

### The 4 Core Subsystems Deployed in Our Codebase:

1. **Central Blackboard DAG & State Store:**
   - Asynchronous, thread-safe blackboard DAG where agents read and write structured claims without direct peer chatter.
   - Dual-persistence layer: Fast In-Memory storage for tests + Redis store with lock leasing.

2. **Formal PEX Protocol & Agent Registry:**
   - Every agent outputs a typed contract: **Prediction (P)** + **Explanation (E)** + **Action Tag (X)** (`RATIFY`, `REVISE`, `REFUTE`, `REJECT`).
   - Automated JSON schema enforcement and self-repair loop to prevent malformed LLM outputs.

3. **Domain Ingestion Adapter (MedAgentBench):**
   - Automated parser that ingests real clinical vignettes, extracts symptoms and lab biomarkers, and dynamically infers and seats the required specialist personas (Cardiologist, Pulmonologist, Arbiter).

4. **Deadlock Attribution & Counterfactual Sandbox Engine:**
   - Algorithmic sliding-window circuit breaker that automatically detects circular rejection loops.
   - Isolated "What-If" sandbox that rolls back state, simulates candidate compromises, and evaluates them with a mathematical scoring function ($S_k$).

> **Presenter Note:** "To make multi-agent reasoning robust and explainable, we engineered four core subsystems: our asynchronous Blackboard DAG, a strict PEX protocol contract that forces agents to provide explanations, an automated benchmark ingestion adapter, and our counterfactual sandbox engine that detects and resolves deadlocks."

---

# Slide 3: Proof of Work #1 — Engineering Verification & Local LLM Telemetry

### How We Know the System is Working: Hard Evidence

1. **100% Automated Test Coverage (179 Passing Tests):**
   - **179 unit tests** pass in **2.2 seconds** across all 4 modules (`pytest -q`).
   - Validates blackboard concurrency, state rollbacks, PEX JSON parsing, scoring formulas, and dataset ingestion.

2. **Real Local LLM Execution (No Mockups, No Paid API Dependencies):**
   - Integrated with local **Ollama** running quantized **Qwen2.5-7B-Instruct** on GPU/CUDA.
   - Real-time token consumption and latency profiling:
     - *Turn 1 (Cardiologist):* $774\text{ tokens} \cdot 2.8\text{s}$ (Proposes CHF)
     - *Turn 2 (Pulmonologist):* $963\text{ tokens} \cdot 3.4\text{s}$ (Refutes with IPF)

3. **Mathematical Convergence Function ($S_k$):**
   - We do not "hope" the models agree; we mathematically verify consensus:
     $$\mathbf{S_k = 0.50\,\text{Agreement} + 0.25\,\Delta\text{Entropy} + 0.15\,\text{Confidence} + 0.10\,\text{Fidelity}}$$
   - Grounded in Claude Shannon’s Information Entropy drop ($\Delta H$) and patient evidence retention.

> **Presenter Note:** "How do we prove this is working? First, we have 179 automated unit tests verifying every single class and contract in 2.2 seconds. Second, our agents run on real local LLMs using Ollama and Qwen2.5-7B with active latency and token tracking. Third, our system uses an information-theoretic scoring function to mathematically prove consensus rather than guessing."

---

# Slide 4: Proof of Work #2 — Live Demonstration & Deadlock Resolution

### Test Case: Stress-Testing on MedAgentBench (`med_task_fibrosis_vs_chf`)
*(Live Visualizer Demo: Dynamic Telemetry Ingestion, Conflict Detection, and Sandbox Self-Healing)*

### What the Demonstration Proves:
1. **Explainable Cognitive Flowcharts:**
   - Expanding any agent node displays its internal 4-stage reasoning:
     *Evidence Ingested $\rightarrow$ Clinical Rule $\rightarrow$ Chain-of-Thought Deliberation $\rightarrow$ Formal PEX Vote*.
2. **Algorithmic Deadlock Detection:**
   - Both specialists refuse to compromise $\rightarrow$ Mutual `REJECT` votes trigger the red **`[DEADLOCK DETECTED]`** circuit-breaker.
3. **The "What-If" Sandbox in Action:**
   - The engine pauses the debate, rolls back to Step 2, and isolates the conflict in a sandbox.
   - Dr. Attending runs 3 iterative candidate trials with peer feedback.
4. **Hill-Climbing Score Trajectory:**
   - Candidate evaluation score progresses: **$0.54$ (Rejected) $\rightarrow 0.78$ (Improving) $\rightarrow 0.95$ (Winner)**.
   - The winning compromise (*IPF primary + secondary cardiac monitoring*) is merged into the live board DAG.

> **Presenter Pitch (Voiceover for Video):**
> *"Here is our live interactive visualizer ingesting real model telemetry from a MedAgentBench case. You can see the cognitive thought flowcharts showing the exact clinical evidence behind every decision. 
> 
> When the Cardiologist and Pulmonologist clash and repeatedly reject each other's claims, our automated deadlock detector flags the impasse. 
> 
> Rather than crashing, our Counterfactual Sandbox rolls back the timeline to Step 2 and runs three simulation trials. As peer feedback is incorporated, our agreement score climbs from 0.54 to 0.95. The winning compromise is verified and merged back to the live board, successfully resolving the conflict."*

---

# Slide 5: Team Contributions & Mid-Semester Milestones

| Team Member | Project Role | Mid-Semester Delivered Milestones |
| :--- | :--- | :--- |
| **Amithav C** | **Blackboard Core & Concurrency** | • Built Blackboard DAG engine, session lifecycle, and turn coordinator<br>• Implemented lock leasing and thread-safe In-Memory and Redis storage<br>• Engineered timeline rollback mechanisms for sandbox branching |
| **Shiva P** | **Agent Personas & Dataset Ingestion** | • Designed and validated specialist personas (Cardiologist, Pulmonologist, Arbiter)<br>• Built MedAgentBench ingestion adapter with automated keyword role inference<br>• Tested dataset prompts against live local Ollama models |
| **Nikhil Palakollu** | **Counterfactual Engine & Sandbox** | • Engineered the sliding-window deadlock detection heuristic<br>• Built the isolated What-If sandbox simulator and prompt generation pipeline<br>• Implemented the 4-pillar mathematical scoring function ($S_k$) and plateau detector |
| **Harish K** | **Visualizer UI & System Evaluation** | • Designed interactive web visualizer with live SVG flowcharts and trace ingestion<br>• Built collapsible cognitive thought flowchart views for each agent<br>• Conducted system performance profiling, token latency benchmarks, and evaluation |

### Next Sprint Goals:
- [ ] Connect multi-model diversity (pairing local Qwen with DeepSeek-R1 / Claude).
- [ ] Implement zero-shot semantic embedding routing for automated specialty discovery.
- [ ] Expand stress-testing to additional benchmark datasets (KramaBench / SWE-bench).

### Thank You!
**Intelligible Multi-Agent Blackboard Architecture**
*We welcome your questions!*

> **Presenter Note:** "In summary, our mid-semester deliverables are fully implemented, verified with 179 green unit tests, and demonstrated on real local model runs. Our next sprint will focus on multi-model diversity and broader benchmark scaling. Thank you, and we are ready for your questions."
