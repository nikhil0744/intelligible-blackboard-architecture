# Intelligible Multi-Agent Blackboard Architecture
## Explainable Collaboration & Conflict Resolution in Multi-Agent AI

---

# Slide 1: Title Slide

### Intelligible Multi-Agent Blackboard Architecture
**Sub-title:** Explainable Collaboration & Self-Healing Conflict Resolution in Multi-Agent AI

**Presented by:**
- Amithav C
- Shiva P
- Nikhil Palakollu
- Harish K

**Focus Areas:** Multi-Agent AI • Clinical Decision Support • System Explainability

> **Presenter Note:** "Good morning everyone. Today we are presenting our project: an Intelligible Multi-Agent Blackboard Architecture that allows specialized AI agents to debate, detect conflicts, and safely resolve complex high-stakes problems."

---

# Slide 2: Project Overview & Our Approach

### The Problem: Single AI Models Have "Tunnel Vision"
- **Black-Box Reasoning:** A single LLM tries to do everything at once, making it hard to trace why it made a decision.
- **Cognitive Blindspots:** In complex fields like medicine, a single model often fixates on one symptom and misses critical warnings or conflicting evidence.

### Our Solution: A Collaborative Multi-Agent Blackboard
- **Shared Digital Blackboard:** Instead of agents talking in a confusing chat loop, they write their findings to a shared, organized board.
- **Clear Communication Rules (PEX):** Every agent must provide three things:
  1. **Prediction:** What they believe is happening.
  2. **Explanation:** Why they believe it (citing specific patient clues).
  3. **Action Tag:** Clear vote (`Agree`, `Propose`, or `Reject`).
- **Smart Specialist Routing:** The system reads the patient case and automatically calls in the right specialists (e.g., Cardiologist for heart symptoms, Pulmonologist for lung symptoms).
- **Automated Deadlock Detection:** If specialists stubbornly reject each other, the system instantly flags a deadlock rather than looping forever.
- **The "What-If" Sandbox:** A senior arbiter agent pauses the debate, steps into an isolated simulation room, and tests compromise solutions until everyone agrees.

> **Presenter Note:** "When a single AI tries to solve a complex patient case, it often gets tunnel vision. Our architecture solves this by using a medical team approach. Specialists post their claims and evidence to a shared digital blackboard. If two specialists hit a stubborn disagreement, our system detects the deadlock, opens an isolated sandbox, tests 'what-if' compromises, and brings a safe, verified consensus back to the team."

---

# Slide 3: Live Video Demonstration

### Case Study: Heart Failure vs. Lung Disease (MedAgentBench)
*(Embedded Screen Recording: Real-Time Telemetry & Conflict Resolution)*

### What Happens in the Demo:
1. **The Debate Begins:**
   - **Cardiologist Agent:** Looks at shortness of breath and fluid sounds $\rightarrow$ Diagnoses Heart Failure.
   - **Pulmonologist Agent:** Looks at the chest CT scan showing lung scarring (honeycombing) and normal heart labs $\rightarrow$ Diagnoses Lung Fibrosis (IPF).
2. **The Impasse:**
   - Both specialists reject each other's diagnosis $\rightarrow$ The red **`[DEADLOCK DETECTED]`** alert triggers!
3. **The "What-If" Sandbox:**
   - The system pauses the debate and rolls back to an earlier step in an isolated sandbox.
   - **Dr. Attending (The Arbiter):** Runs 3 candidate simulation trials to test compromise options with peer feedback.
4. **Agreement Score Climbs:**
   - The solution quality score climbs from **$0.54 \rightarrow 0.78 \rightarrow 0.95$**.
5. **Winning Consensus Merged:**
   - Both specialists agree on a balanced plan: **Treat the lung fibrosis as primary, while keeping secondary cardiac monitoring.**

> **Presenter Pitch (Voiceover for Video):**
> *"Here is a quick demo of our system analyzing a patient case from the MedAgentBench dataset. 
> 
> We have three agents on the board: a Cardiologist, a Pulmonologist, and Dr. Attending who acts as the Arbiter. 
> 
> Initially, both specialists post their diagnoses using structured tags. But because they focus on different symptoms, they enter a deadlock where they keep rejecting each other.
> 
> Watch what happens next: the system detects the deadlock and opens an isolated sandbox. It takes the case back a step and tests three 'What-If' compromise options. As feedback is gathered, our agreement score climbs up to 0.95. The winning compromise is merged back to the main board, where both specialists happily ratify it."*

---

# Slide 4: Team Contributions & Division of Work

| Team Member | Project Role | What They Built |
| :--- | :--- | :--- |
| **Amithav C** | **Blackboard Architecture & Storage** | • Built the central blackboard engine and session coordinator<br>• Implemented fast, thread-safe memory and Redis storage<br>• Designed version tracking so debates can be audited and rewound |
| **Shiva P** | **Dataset Pipeline & Specialist Routing** | • Integrated the MedAgentBench clinical benchmark dataset<br>• Built the keyword scanner that automatically spawns the right specialists<br>• Standardized clinical data contracts between patient records and AI agents |
| **Nikhil Palakollu** | **Agent Protocols & Local AI Models** | • Designed the specialist personas (Cardiologist, Pulmonologist, Arbiter)<br>• Enforced strict JSON output contracts and automatic error-repair<br>• Connected agents to local GPU models (Ollama / Qwen 7B) with latency tracking |
| **Harish K** | **Deadlock Solver & Visualizer Interface** | • Built the automated deadlock detection rules (sliding-window monitor)<br>• Developed the What-If simulation sandbox and agreement scoring engine<br>• Designed the interactive visualizer with live thought flowcharts |

> **Presenter Note:** "Our team divided the project into four clear parts: Amithav built the central blackboard and storage; Shiva handled patient data and automatic specialist routing; Nikhil built the agent personas and connected our local AI models; and Harish engineered the deadlock solver and our interactive visualizer."

---

# Slide 5: Conclusion & Q&A

### Key Takeaways
- **Transparent & Explainable:** Every decision is backed by visible patient clues and clear reasoning steps—no mystery black box.
- **Fail-Safe Conflict Handling:** Disagreements are caught early and resolved through structured compromise rather than endless looping.
- **Teamwork Beats Solo AI:** Diverse specialist agents collaborating on a blackboard make safer, more balanced decisions than any single prompt could.

### Thank You!
**Intelligible Multi-Agent Blackboard Architecture**
*We welcome your questions!*

> **Presenter Note:** "In summary, our architecture proves that giving AI agents a structured workspace to collaborate and resolve disagreements leads to safer and more transparent decisions. Thank you, and we'd love to take your questions."
