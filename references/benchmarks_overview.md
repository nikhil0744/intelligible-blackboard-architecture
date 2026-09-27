# Reference: Evaluation Benchmarks Overview

*Detailed specifications for KramaBench, MSCoRe, and MedAgentBench.*

---

## 1. KramaBench (2025)
- **Domain**: Multi-Agent Information Discovery & Distributed Search.
- **Core Focus**: Measures how effectively autonomous agents coordinate around a central blackboard store to discover non-obvious relationships in unstructured document collections.
- **Relevance**: Tests whether the deterministic scheduler and PXP protocol prevent agents from duplicating search queries or colliding on state writes.

---

## 2. MSCoRe (Multi-Stage Collaborative Reasoning, 2025)
- **Domain**: Complex Industrial Engineering & Science QA.
- **Scale**: Over 126,000 domain-specific QA items across automotive engineering, pharmaceutical chemistry, and energy grid systems.
- **Structure**: Tasks require multiple sequential reasoning stages (e.g. Stage 1: identify thermal threshold $\rightarrow$ Stage 2: compute pressure delta $\rightarrow$ Stage 3: select optimal alloy).
- **Relevance**: Ideal for testing explanation alignment. Disagreements frequently occur in Stage 2 or 3 where conflicting design constraints collide.

---

## 3. MedAgentBench (2025)
- **Domain**: Clinical Reasoning & Multidisciplinary Diagnostic Consensus.
- **Scale**: 100 highly complex clinical cases in v1 (expandable to 300 in v2).
- **Structure**: Simulates clinical panel conferences (tumor boards / diagnostic dilemmas) where multiple specialist agents (Cardiologist, Pulmonologist, Radiologist, Oncologist) must reach unified consensus on patient diagnoses and treatment protocols.
- **Relevance**: The primary benchmark for evaluating the **Deadlock Attribution and Counterfactual Engine**, as clinical cases feature complex differential diagnoses with overlapping, ambiguous symptoms (e.g. heart failure vs. pulmonary fibrosis).
