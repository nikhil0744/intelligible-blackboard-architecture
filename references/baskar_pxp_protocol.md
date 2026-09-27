# Reference: Two-Way Intelligibility Protocol (PXP)

*Based on Baskar et al. & Explainable Multi-Agent Interaction frameworks.*

---

## 1. What is the PXP Protocol?

Traditional AI explanation (XAI) is **one-way**: an AI produces an output and shows an explanation to a passive human observer. 

The **Two-Way Intelligibility Protocol (PXP)** formalizes **agent-to-agent intelligibility**:
- When Agent A speaks to Agent B on the Blackboard, it must explicitly pair its conclusion with its causal rationale.
- Agent B does not merely evaluate the conclusion; it must explicitly evaluate the *explanation* before updating its own internal state.

---

## 2. Core Grammar & Tags

Every assertion on the blackboard is strictly categorized under one of four PXP tags:

| Tag | Formal Meaning | Action on Blackboard |
|---|---|---|
| `PROPOSE` | Independent Thesis | Introduces a new hypothesis or diagnosis with initial supporting evidence. |
| `RATIFY` | Joint Alignment | Concurs with both the **Prediction** ("what") and the **Explanation** ("why"). Reaches consensus. |
| `REVISE` | Belief Modification | Modifies either the prediction or the explanation to synthesize peer evidence and resolve friction. |
| `REFUTE` | Empirical Roadblock | Rejects the target claim based on cited contradictory evidence, but cannot yet propose a substitute. |
| `REJECT` | Total Conflict | Completely disputes both the target prediction and explanation matrix. Triggers deadlock warnings if repeated. |

---

## 3. Mathematical Classification of Intelligibility

The PXP protocol categorizes multi-agent sessions into depth tiers based on how agents interact with explanations:

### Weak Intelligibility
- Consensus is reached through majority voting or authority models.
- Explanations exist in text, but agents do not modify their internal reasoning based on peer explanations.

### Strong Intelligibility
- Consensus is reached where all participating agents ratify both the final prediction $P$ and the joint supporting explanation chain $E_{\text{joint}}$:
  $$\forall a_i \in A, \quad \text{Status}(a_i) = \text{RATIFY}(P, E_{\text{joint}})$$

### Ultra-Strong Intelligibility (The Goal of this Project)
- At least one agent executed a productive `REVISE` transition where its stance was updated **directly as a mathematical consequence of another agent's explanation**:
  $$P_{a_1}^{(t)} \leftarrow f\left(E_{a_2}^{(t-1)}\right) \quad \text{where} \quad P_{a_1}^{(t)} \neq P_{a_1}^{(t-1)}$$
- This proves true collaborative intelligence: mutual belief revision under evidence.
