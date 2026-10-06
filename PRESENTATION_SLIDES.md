# Intelligible Multi-Agent Blackboard Architecture
## A Self-Healing Framework for Collaborative AI Reasoning & Conflict Resolution

---

# Slide 1: Title Slide

### Intelligible Multi-Agent Blackboard Architecture
**Sub-title:** A Self-Healing Framework for Collaborative AI Reasoning & Conflict Resolution

**Presented by:**
- Amithav C
- Shiva P
- Nikhil Palakollu
- Harish K

**Research Area:** Multi-Agent AI Systems • Distributed Coordination • Explainable AI Architecture

> **Presenter Note:** "Good morning everyone. Today we are presenting our project: an Intelligible Multi-Agent Blackboard Architecture—a general-purpose framework that allows specialized AI agents to collaborate, detect circular disagreements, and autonomously self-heal through counterfactual sandboxing."

---

# Slide 2: Project Overview & Our Approach

### The Core Problem: Multi-Agent Systems Get Stuck
- **Black-Box Confusion:** When multiple LLMs talk in open chat loops, conversations become noisy, disorganized, and difficult to audit.
- **Unresolved Impasses:** When models have competing priorities, they either blindly hallucinate a winner or get trapped in circular rejection loops with no way to recover.

### Our Solution: A Self-Healing Blackboard Framework
- **Domain-Agnostic Blackboard DAG:** A central, structured workspace where diverse AI agents post findings asynchronously rather than talking over each other.
- **Strict PEX Communication Protocol:** Every agent communicates using a universal contract:
  1. **Prediction:** The specific claim or recommended action.
  2. **Explanation:** Transparent rationale citing concrete evidence references.
  3. **PXP Action Tag:** A formal vote (`RATIFY`, `REVISE`, `REFUTE`, `REJECT`).
- **Configurable Persona Registry:** The architecture can seat any team of specialist agents—whether for software engineering, financial modeling, or healthcare.
- **Automated Deadlock Detection:** An algorithmic circuit-breaker continuously monitors agent votes and flags a deadlock whenever a circular rejection loop occurs.
- **Counterfactual "What-If" Sandbox:** Automatically pauses the live timeline, branches into an isolated simulation room, rolls back state, and iterates through candidate compromises until consensus is reached.

> **Presenter Note:** "Current multi-agent frameworks often break down when models disagree—they either talk in circles or crash. We engineered an intelligible blackboard architecture. Agents post formal predictions, explanations, and votes to a central board. When the system detects that agents have entered a deadlock, our autonomous Counterfactual Sandbox pauses the debate, steps into an isolated branch, and runs 'What-If' simulations until it finds a verified compromise."

---

# Slide 3: System Evaluation & Live Demonstration

### Stress-Testing the Architecture: The MedAgentBench Benchmark
*(We evaluated our general architecture on MedAgentBench because high-stakes clinical diagnosis provides the ultimate test of competing expert opinions.)*

### What Happens in the Demo:
1. **Specialists at Work:**
   - The system ingests a complex case and seats two competing specialist agents on the board (a **Cardiologist** and a **Pulmonologist**).
   - Each specialist analyzes the problem through their own domain lens and posts contrasting assertions using PEX tags.
2. **The Deadlock Trigger:**
   - Neither agent is willing to back down $\rightarrow$ Both issue mutual `REJECT` votes $\rightarrow$ The blackboard's **`[DEADLOCK DETECTED]`** circuit-breaker triggers!
3. **The "What-If" Sandbox Intervenes:**
   - The system freezes the main branch, rolls back to an earlier decision point, and spawns an isolated simulation sandbox.
   - An **Arbiter Agent** runs 3 candidate "What-If" trials, adjusting prompts and incorporating peer feedback.
4. **Iterative Agreement Scoring:**
   - The consensus score hill-climbs from **$0.54 \rightarrow 0.78 \rightarrow 0.95$**.
5. **Consensus Merged Back to Reality:**
   - The winning balanced compromise is merged back into the live blackboard DAG, and all agents ratify the final decision.

> **Presenter Pitch (Voiceover for Video):**
> *"To prove that our architecture can resolve real-world conflicts, we stress-tested it against a benchmark from MedAgentBench. We chose this benchmark because high-stakes medicine is a domain where specialists naturally clash.
> 
> In this demo, the blackboard seats two specialists with competing views: a Cardiologist and a Pulmonologist. As they debate, they hit a circular rejection loop. 
> 
> Watch how our architecture responds: it immediately flags the deadlock, pauses the debate, and spins up an isolated What-If sandbox. It rolls back the state to Step 2 and runs three simulation trials, gathering peer critique until our agreement score hits 0.95. The winning compromise is then seamlessly merged back to the live board, successfully clearing the impasse."*

---

# Slide 4: Team Contributions & Division of Work

| Team Member | Architectural Pillar | Key Engineering Deliverables |
| :--- | :--- | :--- |
| **Amithav C** | **Core Blackboard Engine & Concurrency** | • Built the central Blackboard DAG, session management, and state store<br>• Implemented lock leasing and thread-safe Redis / In-Memory persistence<br>• Designed the state rollback mechanisms required for timeline branching |
| **Shiva P** | **Benchmark Adapters & Role Routing** | • Built domain adapters for standard benchmarks (MedAgentBench, KramaBench)<br>• Developed automated entity extraction to route domain keywords to personas<br>• Standardized data interchange schemas across heterogeneous test datasets |
| **Nikhil Palakollu** | **Agent Protocols & LLM Infrastructure** | • Engineered the formal PEX protocol schema and automatic JSON error-repair<br>• Built the local LLM model broker (Ollama / Qwen 7B) with concurrency control<br>• Profiled token usage, latency metrics, and agent persona definitions |
| **Harish K** | **Deadlock Detection & Visualizer Telemetry** | • Implemented the sliding-window deadlock detection heuristic<br>• Developed the What-If simulation sandbox and agreement scoring engine ($S_k$)<br>• Built the interactive web visualizer with live SVG flowcharts and trace ingestion |

> **Presenter Note:** "Our team divided the work across the four core pillars of the architecture: Amithav built the underlying blackboard engine and storage; Shiva developed the benchmark adapters and entity-routing pipelines; Nikhil engineered the agent communication protocol and local LLM broker; and Harish built the deadlock detection engine and the interactive telemetry visualizer."

---

# Slide 5: Conclusion & Future Plans

### Current Architecture Strengths
- **Domain-Independent Framework:** Works across software engineering, legal compliance, financial risk, and healthcare.
- **Fail-Safe Conflict Recovery:** Automatic deadlock detection and sandbox simulation stop circular arguments before catastrophic decisions occur.
- **100% Explainable & Auditable:** Every assertion, vote, and state rollback is permanently recorded on the blackboard DAG with full telemetry.

### Future Roadmap & Next Steps
- **Semantic Specialist Routing:** Upgrade from keyword matching to zero-shot semantic embedding routers that automatically decompose any unseen problem into tailored expert personas.
- **Heterogeneous Model Diversity:** Combine different LLM architectures (e.g., DeepSeek-R1 for chain-of-thought math, Qwen for fast critique, and Claude for final arbitration) to prevent shared model biases.
- **Human-in-the-Loop (HITL) Escalation:** If the sandbox engine cannot converge within trial budget ($S_k < 0.70$), automatically escalate to a human expert with the full audit trail and highlighted divergence points.
- **Multi-Benchmark Expansion:** Broaden stress-testing from MedAgentBench to automated software bug-fixing (SWE-bench) and large-scale data engineering (KramaBench).

### Thank You!
**Intelligible Multi-Agent Blackboard Architecture**
*Questions & Discussion*

> **Presenter Note:** "In conclusion, our project establishes a robust, domain-independent multi-agent architecture that turns chaotic LLM arguments into structured, self-healing deliberation. Moving forward, our roadmap includes zero-shot semantic role routing, multi-model diversity to avoid shared blindspots, and human-in-the-loop escalation when automated trials reach an impasse. Thank you, and we'd be thrilled to answer your questions."
