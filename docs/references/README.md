# Literature & Benchmarks Reference Guide

This directory houses research summaries, paper citations, and benchmark specifications for the **Intelligible Blackboard Architecture with Counterfactual Agents**.

---

## Index of Research Documents

| Document | Topic | Key Relevance to Project |
|---|---|---|
| [**`baskar_pxp_protocol.md`**](./baskar_pxp_protocol.md) | Two-Way Intelligibility Protocol (Baskar et al.) | Defines the PXP grammar (`RATIFY`, `REVISE`, `REFUTE`, `REJECT`) and Strong vs. Ultra-Strong Intelligibility metrics. |
| [**`multi_agent_failure_modes.md`**](./multi_agent_failure_modes.md) | Multi-Agent Debate Errors & Failure Distributions | Empirical data on where agentic workflows fail (85% late-stage polarization vs 15% early-stage errors) and backtracking theory. |
| [**`benchmarks_overview.md`**](./benchmarks_overview.md) | KramaBench, MSCoRe, MedAgentBench | Complete breakdown of the 3 evaluation datasets, task counts, domain coverage, and scoring rubrics. |

---

## Foundational Papers & Citations

1. **Two-Way Intelligibility in Multi-Agent Systems**
   - *Authors*: Baskar et al.
   - *Key Insight*: Moves beyond one-way XAI (explaining to a human) to bidirectional agent-to-agent intelligibility, pairing predictions with explanations to enable mutual belief updates.

2. **Improving Factuality and Reasoning in Language Models through Multiagent Debate**
   - *Authors*: Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, Igor Mordatch (2023)
   - *arXiv*: [2305.14325](https://arxiv.org/abs/2305.14325)
   - *Key Insight*: Multi-agent debate improves consensus, but without structured conflict-resolution mechanisms, debates degrade into sycophancy or irreconcilable deadlocks.

3. **Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate**
   - *Authors*: Tian Liang et al. (2023)
   - *arXiv*: [2305.19118](https://arxiv.org/abs/2305.19118)
   - *Key Insight*: Diverse persona prompts foster novel reasoning paths, but require strict coordination to converge.

4. **Why Do Multi-Agent Systems Fail? An Empirical Study of Failure Modes in LLM Agents**
   - *Authors*: Wang et al. (2024)
   - *Key Insight*: Analyzes failure trajectories in multi-agent workflows; proves that errors concentrate in intermediate-to-late synthesis phases due to confirmation bias and unhandled refutations.

5. **KramaBench (2025)**
   - *Focus*: Data and information discovery using multi-agent blackboard systems.

6. **MSCoRe (Multi-Stage Collaborative Reasoning, 2025)**
   - *Focus*: Over 126,000 domain-specific QA items across automotive, pharmaceutical, and energy systems.

7. **MedAgentBench (2025)**
   - *Focus*: 100–300 complex clinical panel reasoning tasks to benchmark diagnostic consensus.
