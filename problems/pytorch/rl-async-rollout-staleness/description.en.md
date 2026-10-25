# RL Systems: Async Rollout Admission, Partial Truncation and Staleness Accounting

## Problem

Synchronous RLHF hurts in a very direct way: within one step, the longest request to generate (a math problem running to 8k tokens) decides the whole step's time, and the remaining cards can only wait. So rollout is broken into an **async service**: the generation side keeps accepting work FCFS and produces samples into a buffer, while the training side publishes a new policy version every `step_time`; once the version updates, the samples sitting in the buffer start to "go stale" (off-policy). AReaL's approach is to accept that "samples are always old" and turn it into a controllable quantity — give every in-flight request a **staleness deadline**, cut it short and deliver it early when the deadline hits (partial rollout), and then correct against a cap $\mu$ (how many versions behind it already is when consumed). verl / OpenRLHF's async / one-step-off-policy path is the same thing under another parameterization.

This problem distills the whole pipeline into one **pure integer event table**: generate no tokens, need no policy model; only, given the data (each request's generation length and arrival time), simulate admission, truncation and batching, then compute staleness, the occupancy timeline and the trainer's idle time.

## Convention

Fix the **version grid**: version $v$ is published at time $v\cdot\text{step\_time}$ and takes effect on the half-open interval $[v\cdot\text{step\_time},(v+1)\cdot\text{step\_time})$. The trainer advances the grid even when the buffer is empty (it idles) — that is exactly what `stall` measures. Every time value is an integer unit (1 unit = generating 1 token).

## Rules

The input is six quantities: `gen_lens` $(N,)$ int64 (the generation length each request needs, $\ge 1$), `arrivals` $(N,)$ int64 (arrival times, **not required to be sorted**), scalar `capacity` $=C\ge1$ (number of concurrent slots, the slots being equivalent), scalar `step_time` $\ge1$, scalar `batch_size` $=B\ge1$, scalar `max_staleness` $=S\ge1$.

1. **Admission (FCFS)**: the queue order is lexicographic by $(\text{arrivals},\ \text{request id})$. Requests take work one by one in that order, with start time
   $$\text{start}_i=\max(\text{arrivals}_i,\ \text{the time the earliest slot frees})$$
   the slots' free times initialized to $0$; a request is given to "the slot that frees earliest" (ties → smallest slot id, and since slots are equivalent this affects no result), and that slot's next free time is set to this request's `finish`.
2. **Partial rollout deadline**: entry version $v0_i=\lfloor \text{start}_i/\text{step\_time}\rfloor$,
   $$\text{deadline}_i=(v0_i+S)\cdot\text{step\_time},\quad
     \text{finish}_i=\min(\text{start}_i+\text{gen\_len}_i,\ \text{deadline}_i)$$
   `truncated[i]` $=1$ if and only if $\text{finish}_i<\text{start}_i+\text{gen\_len}_i$ — **finishing exactly on the deadline does not count as truncation**. $S\ge1$ guarantees $\text{finish}_i\ge \text{start}_i+1$ (because $\text{start}_i-v0_i\cdot\text{step\_time}\le \text{step\_time}-1$), so no zero-length request ever appears.
3. **Batch by completion order**: queue the requests lexicographically by $(\text{finish},\ \text{request id})$; the request at rank $q$ goes into `batch_id` $=\lfloor q/B\rfloor$ (the last batch may hold fewer than $B$). The whole batch **shares** one consumption version: $c_k=\lfloor \text{ready}_k/\text{step\_time}\rfloor$, where $\text{ready}_k$ is the latest `finish` in the batch (the source of the short-plank effect).
4. **Staleness**: `mu[i]` $=c_{\text{batch\_id}[i]}-v0_i$, i.e. how many versions separate "the policy that was in force when this sample started generating" from "the policy when it was consumed". It always holds $\text{mu}\ge0$ ($\text{ready}\ge\text{finish}\ge\text{start}$).
5. **Occupancy timeline**: `occupancy[tau]` $=\#\{i:\text{start}_i\le \text{tau}<\text{finish}_i\}$ for $\text{tau}=0,1,\dots,\max_i\text{finish}_i-1$, so its length is just the latest finish time (endpoint excluded).
6. **Idle time**: `stall[k]` $=\max\!\big(0,\ (c_k-k)\cdot\text{step\_time}\big)$, of length $\lceil N/B\rceil$. Under an ideal pipeline the $k$-th batch is consumed exactly at version $k$; when generation cannot keep up a positive stall appears, and when supply runs over it is clamped to $0$ (the version grid never moves backwards).

Return a five-tuple (all int64): `mu` $(N,)$, `batch_id` $(N,)$, `truncated` $(N,)$, `occupancy` $(\max\text{finish},)$, `stall` $(\lceil N/B\rceil,)$. For $N=0$ all five are empty tensors of length $0$.

## Example

$N=8$, $C=2$, $\text{step\_time}=4$, $B=3$, $S=2$, `gen_lens=[6,7,9,5,4,8,3,10]`, `arrivals=[0,0,1,2,5,6,10,11]`. The intermediate table (worked out step by step from the rules above, for self-checking):

```
 i  arr gen start v0 deadline finish trunc |  bid ready c  mu
 0   0   6     0   0     8      6     0 |   0   12 3   3
 1   0   7     0   0     8      7     0 |   0   12 3   3
 2   1   9     6   1    12     12     1 |   0   12 3   2
 3   2   5     7   1    12     12     0 |   1   19 4   3
 4   5   4    12   3    20     16     0 |   1   19 4   1
 5   6   8    12   3    20     20     0 |   2   24 6   3
 6  10   3    16   4    24     19     0 |   1   19 4   0
 7  11  10    19   4    24     24     1 |   2   24 6   2
```

```
mu        = [3, 3, 2, 3, 1, 3, 0, 2]
batch_id  = [0, 0, 0, 1, 1, 2, 1, 2]
truncated = [0, 0, 1, 0, 0, 0, 0, 1]
occupancy = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1]   # length 24 = latest finish
stall     = [12, 12, 16]        # consumption versions [3,4,6] vs ideal [0,1,2]
```

A few readings: request 3 should finish at $7+5=12$ and its deadline is also $12$; **equal ⇒ not truncated**. Requests 5 and 7 both land in batch 2, are dragged out to $\text{ready}=24$ by request 7, so request 5 — which would otherwise have lagged by only one version — also becomes $\mu=3$. Both truncated requests satisfy $\mu\ge S=2$.

The same data with only $S$ changed (everything else fixed):

```
S=1  → truncated=[1,1,1,1,0,1,0,1]  mu=[2,2,1,2,1,1,1,1]  batch_id=[0,0,0,1,1,1,2,2]
       stall=[8,8,8]   occupancy length 16 = [2]*15+[1]
S=10 → truncated all 0  mu=[3,3,3,2,1,4,0,3]  batch_id=[0,0,1,0,1,2,1,2]
       stall=[12,12,20]  occupancy length 29 = [2]*23+[1]*6
```

Change $S$, change the truncation, and **the completion order changes, so batch membership changes too** (request 2 moves from batch 0 to batch 1); even the idle time shifts. That is the "trade truncation for freshness" cost accounting.

## Constraints

- CPU; $0\le N\le 256$, $1\le C\le 64$, $1\le \text{step\_time}\le 64$, $1\le B\le 256$, $1\le S\le 32$; `gen_lens` $\ge1$, `arrivals\ge0$; all integer arithmetic, no floating-point error.
- Judgement: all five outputs compared with `torch.equal` (int64).
- Degenerate invariants: $C\ge N$ with $S$ large enough $\Rightarrow$ `start` equals arrival and `occupancy` peak $=N$; $C=1\Rightarrow$ `occupancy` all $1$ (serial, length $=\sum$ lengths); $B\ge N\Rightarrow$ only batch 0, `stall` length $1$; $S$ very large $\Rightarrow$ `truncated` all $0$; ample supply $\Rightarrow$ `stall` all $0$; a truncated sample always has $\text{mu}\ge S$.
- Division of labour: `inf-continuous-batching-scheduler` only computes "at what time does each request finish after FCFS completion", with no version axis, no truncation, no consumption-side batching; this problem reuses its admission core and adds the **deadline feeding back into the timeline**, **completion-order batching**, and **staleness/idle accounting**. How $\mu$ then enters the loss (off-policy correction) belongs to `to-ppo-clipped-surrogate` / `to-kl-controller`; this problem only gets $\mu$ right.
