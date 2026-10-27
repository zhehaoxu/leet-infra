# Multi-LoRA Hot Swap: Adapter Residency LRU and Memory Accounting

## Problem

When one base model serves dozens or hundreds of LoRA adapters, they cannot all live in GPU memory. The adapter manager in S-LoRA / Punica / vLLM runs a **bounded residency pool**: a request arrives carrying an adapter id; if that adapter is resident it is used directly (a hit), otherwise it is swapped in (a miss), and when the pool is full the least recently used adapter is swapped out.

Two engineering details separate this from textbook LRU:

- **pin**: a few hot adapters are pinned resident and never swapped out. They consume a fixed share, and only what remains is under LRU control.
- **rejection**: if pinned adapters fill the pool and the requested adapter is not there, the request can only be rejected — queueing it until a pinned adapter disappears is pointless, it never will.

Run that state machine and report hits, eviction sequence, load count, rejection count, and peak memory.

## Task

Implement `solve(requests, adapter_bytes, capacity, pinned)`:

- Inputs: `requests` `(M,)` adapter ids in arrival order; `adapter_bytes` `(A,)` memory per adapter; `capacity` residency limit; `pinned` `(P,)` pinned adapters (already loaded at the start).
- For each request `a`:
  - `a` resident → hit, refresh its recency;
  - otherwise a miss: load directly if the pool has room; if full, evict the least recently used **non-pinned** adapter and then load;
  - pool full with only pinned tenants → count it in `rejected`, change nothing.
- Outputs: `hit` `(M,)` bool, `evicted` `(M,)` int64 (adapter evicted this step, -1 if none), `loads` int64 (dynamic loads, excluding the initial pinned set), `rejected` int64, `peak_bytes` int64 (peak sum of resident adapter bytes, pinned included).

## Example

```
adapter_bytes = [100, 200, 300, 400]
capacity = 2, pinned = [0], requests = [1, 1, 2, 2, 1]

initial residency = {0} (pinned), 100 bytes

r0: adapter 1  miss -> load, resident {0,1}   loads=1  peak=300
r1: adapter 1  hit                            hit
r2: adapter 2  miss -> evict LRU=1 (the only non-pinned one), load 2   evicted=1  loads=2  peak=400
r3: adapter 2  hit
r4: adapter 1  miss -> evict 2, load 1                                evicted=2  loads=3  peak=400

hit     = [F, T, F, T, F]
evicted = [-1, -1, 1, -1, 2]
loads   = 3
rejected = 0
peak_bytes = 400
```

## Constraints

- **Pinned adapters never leave**: they start resident, sit outside the LRU queue, and are never evicted.
- **A hit must refresh recency**: without it the policy is FIFO and evicts the wrong adapter on sequences like `1,2,1,3`.
- **The eviction target is the least recently used non-pinned adapter**, not the least recently used adapter overall — including pinned ones in the candidates throws a pinned adapter out.
- **Rejection is terminal**: a rejected request is not queued and changes nothing; it only increments `rejected`.
- **`peak_bytes` includes the initial pinned occupancy**, so the peak at step 0 is not 0.
- **`pinned` must be duplicate-free, in range, and no larger than `capacity`**; otherwise raise `ValueError`.
