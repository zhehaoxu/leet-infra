# Approach

## Interviewer's Intent

"Why does pipeline parallelism require the number of micro-batches to be much larger than the number of stages?" "Why do p=4, m=4 give only 4/7 utilization?" — the bubble ratio $\frac{p-1}{m+p-1}$ is the first-principles formula for answering such questions. It captures the core trade-off of the PP dimension among the four-dimensional parallelism of large-model training (DP/TP/PP/EP), and is almost always asked.

## Derivation

Let one micro-batch passing through one stage count as 1 unit. GPipe arranges the forwards along a diagonal: the `i`-th micro-batch occupies time slot `i+j` at stage `j`. Hence:
- The latest finishing slot is $(m-1)+(p-1)$, i.e. makespan $T = m+p-1$.
- Each stage is busy for $m$ slots (processing $m$ micro-batches), and the remaining $T-m=p-1$ slots are bubbles.
- Utilization $=m/T$, bubble ratio $=(p-1)/T$.

With symmetric fwd/bwd (each taking the same time one after the other), the numerator and denominator are both multiplied by 2 and cancel, so the ratio depends only on $m,p$ — this is the classic result "bubbles depend only on the number of stages and the number of micro-batches".

## Key Intuition

$$ \text{bubble}=\frac{p-1}{m+p-1}\ \Rightarrow\ \text{to shrink bubbles you need } m\gg p $$

- With $m$ fixed, a larger $p$ → larger bubbles (diminishing returns; this is why PP usually stays within dozens of stages on a single node).
- To use more stages while keeping efficiency, the global batch must be split into more micro-batches (limited by memory and the effective batch size).

## Relationship to Other Problems

This problem computes only **bubbles/time**; the memory dimension is covered by the 1F1B problem. The communication dimension: PP transfers activations only between adjacent stages (point-to-point, a very small volume), a different bottleneck source from All-Reduce-type collectives (`parallel-allreduce`, `distributed-zero-communication`) — this is one piece of the overall picture "PP saves layer memory at the cost of bubbles, while TP/DP cost bandwidth".

## Implementation and Grading

Three lines of closed form suffice. `test_cases.py` uses an independent simulation that **fills the busy cells diagonally** as the oracle (not copying the formula directly), and verifies `bubble+util==1`, the $p=1$/$m=1$ boundaries, and that bubbles decrease monotonically as $m$ grows.
