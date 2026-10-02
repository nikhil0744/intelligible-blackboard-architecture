# Reference: Multi-Agent Failure Modes & Empirical Error Distribution

*Synthesized from Du et al. (2023), Wang et al. (2024), and Liang et al. (2023).*

---

## 1. Where Multi-Agent Systems Fail (Empirical Distribution)

Multiple empirical studies analyzing multi-agent reasoning trajectories reveal that errors are **not evenly distributed** across a debate. They are heavily concentrated in the **later synthesis phases**:

```
Debate Progression Timeline:
[Step 0: Context] ────────► [Early Turns: Facts] ────────► [Late Turns: Resolution]
       0%                          10%–15%                          85%–90%
  (Problem Input)              (Initial Extraction)             (Polarization & Deadlock)
```

### Empirical Statistics:
- **10%–15% of Errors occur in Steps 1–2**:
  Modern LLMs (even 7B/8B models) reliably extract basic parameters from prompts (patient age, numerical values, clear constraints).
- **85%–90% of Errors occur in Steps 3+**:
  Errors arise from:
  1. **Confirmation Bias**: An agent over-indexes on its own initial hypothesis and rejects valid counter-evidence.
  2. **Adversarial Polarization**: When Agent A says `REFUTE`, Agent B perceives it as an attack and escalates with `REJECT`, triggering cyclic deadlocks.
  3. **Context Thrashing**: Without a shared blackboard, linear message passing causes agents to forget constraints established in Step 1.

---

## 2. Backtracking Search Strategies: Bounded vs. Binary Search

When resolving deadlocks, the search strategy over historical checkpoints ($C_0, C_1, \dots, C_N$) must match the underlying probability distribution:

### Why Backward Sliding Window ($N \rightarrow N-1 \rightarrow N-2$) is Optimal:
1. **Geometric Distribution of Root Causes**:
   Because ~85% of errors originate in the most recent 1 to 2 turns, the probability that the error is at the latest checkpoint $C_N$ is **$P \approx 0.65$**.
   - Searching $C_N$ first solves **65% of cases on Attempt 1** in $O(1)$ time.
   - Searching $C_{N-1}$ second solves another **20% of cases on Attempt 2**.
   - **Expected search steps**: $E[\text{steps}] = 1 \cdot 0.65 + 2 \cdot 0.20 + 3 \cdot 0.08 \approx 1.29 \text{ attempts}$!

2. **Why Binary Search Fails in Multi-Agent Checkpoints**:
   - Binary search ($O(\log N)$) assumes that checking an item is cheap and that the error property is monotonic.
   - In a multi-agent sandbox, "testing" a checkpoint requires **re-generating multiple agent debate turns forward**.
   - If a debate has 4 checkpoints, binary search jumping to $C_2$ **immediately throws away Checkpoints 3 and 4**, forcing expensive full re-simulation even though the error was almost certainly in Checkpoint 4!
   - For typical debate lengths ($N \le 5$ checkpoints), a backward linear search with depth cap $K=3$ has a significantly lower expected token cost than bisection.
