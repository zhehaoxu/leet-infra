# Step-by-Step Ring All-Reduce Communication Simulation

## Problem

`distributed-all-reduce` only asks for the final value. This problem asks you to **reproduce every communication step of Ring All-Reduce exactly**, returning the **global intermediate state** after each communication round. This is the algorithm NCCL uses to All-Reduce gradients in real clusters — breaking "just summing" apart into a visible flow of chunks.

## Data Layout

The $W$ ranks are arranged in a ring, and the **right neighbor** of rank `q` is $(q+1)\bmod W$. Each rank holds a buffer of length $W$, whose `c`-th column is its data for the **`c`-th chunk**. The input `init` is a tensor of shape $(W, W)$, where row `q` = rank `q`'s initial buffer. Goal: every rank ends up holding the complete vector of per-chunk sums $\text{full}[c]=\sum_q \texttt{init}[q][c]$.

## Communication Schedule (implement exactly as specified)

**Each step is one synchronous communication round**: all sends in the round are computed from the buffer **before** the update; each rank sends exactly one chunk to its right neighbor and receives exactly one chunk from its left neighbor.

**Phase 1 · Reduce-Scatter** ($t = 0 \dots W-2$, $W-1$ steps in total):
rank `q` sends the value `buf[q][s]` at index $s=(q-1-t)\bmod W$ to its right neighbor; the receiver **accumulates** it into its own buffer at the **same index `s`**. After $W-1$ steps, index `q` of rank `q` holds exactly the complete sum (the other chunks are still partial).

**Phase 2 · All-Gather** ($t = 0 \dots W-2$, $W-1$ steps in total):
rank `q` sends the value `buf[q][s]` at index $s=(q-t)\bmod W$ to its right neighbor; the receiver **overwrites** its own buffer at the **same index `s`** with it (it is already the complete sum, so just move it). After $W-1$ steps every rank's buffer equals `full`.

## Rules

Implement `solve(init)`: `init` is a $(W,W)$ tensor. Execute the schedule above and return a **list of snapshots** — after each completed communication step, record "the buffers of all ranks at that moment", shape $(W,W)$. Return `list[torch.Tensor]` with **length $=2(W-1)$**, ordered as the $W-1$ Reduce-Scatter snapshots followed by the $W-1$ All-Gather snapshots. Do not modify the input `init`. For $W=1$ return the empty list `[]`.

## Examples ($W=3$, complete trace)

`init = [[1,2,3],[4,5,6],[7,8,9]]`, `full = [12,15,18]`:

```
step0  RS t=0 : [[1,10,3],[4,5,9],[11,8,9]]
step1  RS t=1 : [[12,10,3],[4,15,9],[11,8,18]]   # diagonal buf[q][q] already holds the complete sum
step2  AG t=0 : [[12,10,18],[12,15,9],[11,15,18]]
step3  AG t=1 : [[12,15,18],[12,15,18],[12,15,18]]  # everyone has full
```

Return the list of these 4 $(3,3)$ tensors.

## Constraints

- `1 ≤ W ≤ 8`; `init` is $(W,W)$
- Return `list[torch.Tensor]` of length $=2(W-1)$, each of shape $(W,W)$, compared element-wise
- Implement strictly according to the index formulas in the statement; different valid ring arrangements produce different intermediate states, and grading recognizes only this schedule
