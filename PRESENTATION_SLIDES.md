# Intelligible Multi-Agent Blackboard Architecture
## Deliberative Collaboration, Conflict Detection & Counterfactual Resolution in Complex Domains

---

# Slide 1: Title Slide

### Intelligible Multi-Agent Blackboard Architecture
**Sub-title:** Deliberative Collaboration, Conflict Detection & Counterfactual Resolution in Complex Domains

**Presented by:**
- Amithav C
- Shiva P
- Nikhil Palakollu
- Harish K

**Domain:** Multi-Agent AI Systems • Healthcare Decision Support • LLM Reasoning

> **Presenter Note:** "Good morning everyone. Today we present our project: an Intelligible Multi-Agent Blackboard Architecture for collaborative, explainable reasoning in complex, high-stakes domains."

---

# Slide 2: Project Overview & Our Approach

### The Challenge: Single-Agent LLM Failures
- **Cognitive Blindspots:** Monolithic LLMs act as unexplainable "black boxes" prone to hallucinations in multi-constraint problems.
- **Specialty Bias:** In complex domains like medicine, single models suffer from tunnel vision, optimizing for one symptom while missing critical contraindications.

### Our Solution: Multi-Agent Blackboard Architecture
- **Shared Blackboard DAG:** A decentralized workspace where specialist agents collaborate asynchronously without noisy peer-to-peer chatter.
- **Strict PEX Protocol Contract:** Every contribution outputs formal **Prediction (P)** + **Explanation (E)** + **PXP Tag** (`RATIFY`, `REVISE`, `REFUTE`, `REJECT`).
- **Dynamic Persona Ingestion:** Automated specialty inference engine (`MedAgentBenchAdapter`) that extracts clinical symptoms from patient vignettes and spawns corresponding specialist personas (e.g., Cardiologist, Pulmonologist).
- **Automated Deadlock Detection:** Real-time sliding-window monitoring flags circular rejections and deadlocks when specialists reach an impasse.
- **Autonomous Counterfactual Sandbox:** Spins up an isolated sandbox, rolls back the timeline, simulates "What-If" compromise trials, and merges a Pareto-optimal resolution back to reality.

> **Presenter Note:** "When single LLMs handle complex dilemmas, they often suffer from tunnel vision. To solve this, we built a collaborative blackboard architecture. Specialist agents post structured predictions and explanations to a shared board. Our system continuously monitors their dialogue: when specialists enter a circular deadlock, our Counterfactual Sandbox isolates the conflict, tests candidate compromise solutions, and merges the winning resolution back to the main timeline."

---

# Slide 3: Live Video Demonstration

### Case Study: High-Stakes Pulmonology vs. Cardiology Dilemma (MedAgentBench)
*(Embedded Video: Real-Time Visualizer Telemetry & Counterfactual Deadlock Resolution)*

### Key Architecture Highlights Shown in Demo:
1. **Multi-Disciplinary Specialist Panel:**
   - **Dr. Cardiologist** (focuses on dyspnea & bibasilar crackles $\rightarrow$ proposes Congestive Heart Failure).
   - **Dr. Pulmonologist** (focuses on HRCT honeycombing & normal BNP $\rightarrow$ refutes with Idiopathic Pulmonary Fibrosis).
2. **Deadlock Collision Trigger:**
   - Both specialists issue mutual `REJECT` tags, triggering the red **`[DEADLOCK DETECTED]`** circuit-breaker.
3. **Isolated "What-If" Counterfactual Sandbox:**
   - System rolls back to Step 2 in an isolated branch.
   - **Dr. Attending (Counterfactual Arbiter)** runs 3 iterative candidate trials with peer feedback.
4. **Hill-Climbing Agreement Scoring ($S_k$):**
   - Candidate evaluation score progresses from **$0.54 \rightarrow 0.78 \rightarrow 0.95$**.
5. **Pareto-Optimal Consensus Merge:**
   - Unified care plan merged into the main DAG: *"Idiopathic Pulmonary Fibrosis (primary) with secondary cardiac monitoring."*

> **Presenter Pitch (Voiceover for Video):**
> *"Here is a live demonstration of our system using a clinical case from the MedAgentBench dataset. On the blackboard, we have two medical specialists—a Cardiologist and a Pulmonologist—along with Dr. Attending serving as the Counterfactual Arbiter.*
>
> *Initially, both specialists analyze the patient's symptoms and post their own diagnostic claims using our PXP protocol tags. However, because each specialist focuses on different symptoms, they enter a circular REJECT loop.*
>
> *As soon as this deadlock is detected, the system spawns an isolated sandbox, rolls back to Step 2, and runs three 'What-If' simulation trials against peer critiques. Our agreement scoring function evaluates peer convergence: the highest-scoring Pareto-optimal solution ($S_k = 0.95$) is injected back into the main blackboard timeline, where the specialists ratify it and conclude the debate with a unified diagnosis."*

---

# Slide 4: Team Contributions & Division of Work

| Team Member | Project Pillar | Core Architectural Deliverables |
| :--- | :--- | :--- |
| **Amithav C** | **Blackboard Core & Concurrency** | • Asynchronous Blackboard DAG engine & session lifecycle<br>• Lock leasing, conflict isolation, and Redis / In-Memory storage backends<br>• Strict versioning and audit-trail persistence |
| **Shiva P** | **Dataset Pipelines & Role Ingestion** | • MedAgentBench & KramaBench dataset ingestion adapters<br>• Automated symptom keyword extraction & specialty inference (`infer_clinical_specialties`)<br>• Standardized clinical contract data models |
| **Nikhil Palakollu** | **Agent Panel & PEX Protocol** | • Multi-agent persona registry (Cardiologist, Pulmonologist, Attending)<br>• PEX schema contract enforcement, JSON retry repairs, and local Ollama GPU broker<br>• Token telemetry and inference latency profiling |
| **Harish K** | **Counterfactual Engine & Visualizer** | • Automated sliding-window deadlock detection heuristic<br>• Isolated What-If sandbox manager & hill-climbing scoring engine ($S_k$)<br>• Interactive web visualizer with cognitive flowcharts & live trace ingestion |

> **Presenter Note:** "Our team divided the project into four cohesive engineering pillars: Amithav built the concurrent blackboard DAG and storage layer; Shiva developed the MedAgentBench ingestion adapters and automated role inference; Nikhil engineered the multi-agent PEX protocol and local Ollama GPU broker; and Harish implemented the counterfactual deadlock solver and the real-time intelligible visualizer."

---

# Slide 5: Conclusion & Q&A

### Summary of Impact
- **From Black-Box to Auditable AI:** Every assertion is grounded in evidence references, explicit clinical rules, and verifiable PXP tags.
- **Fail-Safe Conflict Governance:** Autonomous deadlock detection stops circular debates before catastrophic or biased decisions are made.
- **Self-Healing Synthesis:** The Counterfactual Sandbox discovers Pareto-optimal compromise solutions that single LLMs miss.

### Thank You!
**Intelligible Multi-Agent Blackboard Architecture**
*Questions & Discussion*

> **Presenter Note:** "In summary, our system turns multi-agent LLM reasoning from an unpredictable black box into a transparent, self-correcting, and production-grade architecture. Thank you, and we are now open to take your questions."
