# Solution Approach

## Interviewer's Intent

Many people mistakenly believe that ZeRO "saves both memory and communication." The opposite is true: ZeRO-1/2 communicates **exactly the same** as DDP, while ZeRO-3 communicates
**more**. Being able to explain "what extra movement partitioning introduces" is the key to distinguishing "has used ZeRO" from "understands ZeRO." This problem quantifies
this tradeoff with a single simplified model.

## Where the Ring Coefficient c Comes From

Every GPU's send and receive in ring-style primitives carries $c=\frac{W-1}{W}$: one All-Reduce (= Reduce-Scatter + All-Gather) moves
$2cS$ per GPU; doing only one of the two directions costs $cS$. When $W$ is large, $c\to 1$, so "All-Reduce ≈ each GPU moves the data twice" is a convenient
approximation.

## Breaking Down the Three Schemes Item by Item

| Scheme | Communication composition | Bytes per GPU |
|---|---|---|
| DDP | Gradient All-Reduce | $2cG$ |
| ZeRO-1/2 | Gradient Reduce-Scatter + updated-parameter All-Gather | $cG + cG = 2cG$ |
| ZeRO-3 | The row above + one parameter All-Gather each in forward/backward | $2cG + 2cP$ |

- ZeRO-1/2 only changes "where optimizer states/gradients live"; gradient synchronization + parameter broadcast together still equal one All-Reduce,
  so it ties with DDP.
- ZeRO-3 also stores the parameters themselves in shards, so the forward pass must temporarily gather the layer's parameters and the backward pass gathers them once more, moving an extra $2cP$.

## Implementation Notes

Compute $c$ first (note that $W=1$ gives $c=0$ → all zeros), then $G=N\cdot\texttt{grad\_bytes}$, $P=N\cdot\texttt{param\_bytes}$,
and combine them according to the table above. `check_zero_equals_ddp` and `check_zero3_heavier` test whether these two counterintuitive conclusions are actually implemented correctly.

## Relation to the Memory Problem

`training-state-memory` gives the **per-GPU memory** of ZeRO-1/2/3 ($16N \to 16N/W$); this problem gives the corresponding **per-GPU communication**.
Together, the two problems form the complete ZeRO tradeoff: **trading communication for memory**—the higher the stage, the more memory is saved and the more expensive the communication.

## Judging Criteria

`test_cases.py` cross-checks against an independent reference `_ref` that accumulates step by step, and verifies ZeRO-1/2 ≡ DDP, ZeRO-3 strictly more expensive, $W=1$ no communication,
and linear doubling of bytes with dtype width.
