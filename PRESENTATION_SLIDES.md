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

| Team Member | Project Role | Core Engineering Contributions |
| :--- | :--- | :--- |
| **Shiva P** | **Agent Personas & Live Testing** | • Designed and configured the specialist agent personas (Cardiologist, Pulmonologist, Arbiter)<br>• Tested dataset test cases and prompt variations on live local models (Ollama / Qwen 7B)<br>• Validated agent response consistency and PEX protocol adherence |
| **Amithav C** | **Blackboard Architecture & Scheduler** | • Designed and implemented the core Blackboard DAG architecture<br>• Built the agent scheduling logic, session lifecycle, and turn-taking coordinator<br>• Implemented concurrency control, lock leasing, and state storage engines |
| **Nikhil Palakollu** | **Counterfactual Logic & Sandbox Simulator** | • Engineered the entire counterfactual agent decision logic and conflict attribution<br>• Built the isolated What-If sandbox simulator and state rollback mechanisms<br>• Developed the iterative hill-climbing agreement scoring function ($S_k$) |
| **Harish K** | **Visualizer Presentation UI & Evaluation** | • Designed and developed the interactive presentation UI/visualizer with live SVG flowcharts<br>• Implemented real-time telemetry trace ingestion and collapsible cognitive thought views<br>• Conducted system performance evaluation, latency profiling, and benchmark analysis |

> **Presenter Note:** "Our team divided the work into four clear areas: Shiva designed the specialist agent personas and ran tests on our live models; Amithav built the central blackboard architecture and scheduler; Nikhil developed the counterfactual agent logic and the What-If sandbox simulator; and Harish built the interactive presentation UI and led the system evaluation."

---

# Slide 5: Conclusion & Future Plans

### Key Architecture Strengths
- **Works in Any Domain:** General-purpose design suited for medicine, software engineering, law, or finance.
- **Stops Infinite Loops:** Automated deadlock detection and sandbox simulation catch and resolve disagreements safely.
- **Clear & Explainable:** Every claim, vote, and state rollback is visible and auditable—no black box.

### Future Plans (Simple Roadmap)
1. **Smarter Expert Routing:** Automatically detect and assign the right specialist agents from raw text instead of relying on keywords.
2. **Mixing Different AI Models:** Combine different LLMs (e.g., Claude, GPT, and open-source models) so agents don't share the same blindspots.
3. **Human-in-the-Loop Backup:** If agents ever reach an unresolvable impasse, automatically hand the case to a human expert with a complete summary of the debate.
4. **Testing More Fields:** Expand evaluation into legal contract review, financial risk analysis, and software code debugging.

### Thank You!
**Intelligible Multi-Agent Blackboard Architecture**
*Questions & Discussion*

> **Presenter Note:** "In summary, our architecture gives AI agents a structured workspace to collaborate, detect conflicts, and self-heal through sandboxed simulations. Our future plans focus on smarter agent selection, mixing different AI models to eliminate shared biases, and adding a human-in-the-loop fallback for unresolved debates. Thank you, and we are now open for questions!"
