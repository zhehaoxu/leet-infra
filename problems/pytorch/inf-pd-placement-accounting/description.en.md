# Placement Accounting for Disaggregated Prefill/Decode: KV Handoff, TP Ring Replication, and Memory Admission

## Problem Statement

Once prefill and decode are split onto separate instances (disaggregation), the server inherits a cost that did not exist before: **the KV computed by prefill must be moved onto some decode replica**. That cost is constrained by three things at once — when the prefill pool finishes, when the KV transfer link becomes free, and how much memory the target replica has left.

Given a batch of requests, several decode replicas (each with its own capacity, link bandwidth, and tensor-parallel degree), and a prefill-pool throughput, simulate the placement pass: which replica each request lands on, at which tick its KV is fully resident, how many bytes each link moved in total, how much memory each replica holds, and how many requests were dropped because no replica could take them.

Topology and assumptions:

- All requests are prefilled one after another in index order (FCFS, non-overlapping), with the pool processing `prefill_tps` prompt tokens per tick.
- After prefill finishes, the KV is moved over that replica's transfer link. **Transfer does not overlap prefill compute**, and each link moves one request at a time, so later requests queue.
- If a decode replica runs tensor parallelism, the KV also needs one ring all-gather inside the replica (so every TP rank holds the full KV); that traffic is charged to the same link at the "(C−1)/C per-link share" convention.
- A request's memory footprint is its peak: KV for the prompt plus KV for the tokens it will generate.

## Rules

1. **Prefill completion tick**: let `cum_prompt[i]` be the sum of `prompt_lens[0..i]`; then

   ```
   prefill_done[i] = ceil(cum_prompt[i] / prefill_tps)
   ```

   Take the ceiling **on the cumulative total**, not per request.

2. **KV bytes of a request** (peak):

   ```
   kv[i] = (prompt_lens[i] + gen_lens[i]) * kv_bytes_per_token
   ```

3. **Bytes on one link**: for decode replica `d`,

   ```
   ring   = ceil(kv[i] * (tps[d] - 1) / tps[d])      # ring all-gather's per-link share
   moved  = kv[i] + ring                              # ring == 0 when tps[d] == 1
   ```

4. **Placement, one request at a time** (in increasing index order; each decision takes effect immediately):

   - Memory admission first: keep only replicas with `resident[d] + kv[i] <= caps[d]` (equality still fits).
   - Estimate the completion tick on every admissible replica:

     ```
     start  = max(prefill_done[i], link_free[d])
     finish = start + ceil(moved / bws[d])
     ```

     where `link_free[d]` is the completion tick of the previous transfer on that link (initially 0).
   - Pick the replica with the smallest `finish`; ties go to the smaller index. Note this is **not** "least loaded" or "widest link".
   - On success: `resident[d] += kv[i]`, `wire[d] += moved`, `link_free[d] = finish`.

5. **Dropping**: if no replica passes admission, the request is dropped — `placement` and `ttft` become `-1`, it occupies neither memory nor link, but its prefill share is **already spent**, so later requests' `prefill_done` still counts it.

6. **TTFT** is defined as the tick at which the KV is fully resident, i.e. when the request's first token can be produced.

Return a 6-tuple:

- `placement`: which decode replica each request landed on; `-1` if dropped;
- `ttft`: the KV-resident tick of each request; `-1` if dropped;
- `wire`: total bytes moved on each replica's link (including the ring share);
- `resident`: total KV bytes of admitted requests on each replica;
- `rejected`: number of dropped requests;
- `makespan`: the largest `ttft` among admitted requests; 0 if none were admitted.

## Example

`prompt_lens = [20, 30, 100, 10]`, `gen_lens = [30, 10, 100, 5]`, `kv_bytes_per_token = 4`, `caps = [1000, 1000]`, `bws = [64, 32]`, `tps = [1, 2]`, `prefill_tps = 10`

Prefill completion ticks (cumulative 20/50/150/160): `2 / 5 / 15 / 16`.

```
Request 0: kv = (20+30)*4 = 200
  replica 0 (tp=1): moved = 200, start = 2, finish = 2 + ceil(200/64) = 6
  replica 1 (tp=2): ring = ceil(200*1/2) = 100, moved = 300, finish = 2 + ceil(300/32) = 12
  -> replica 0, ttft = 6; resident[0] = 200

Request 1: kv = (30+10)*4 = 160
  replica 0: start = max(5, 6) = 6, finish = 6 + ceil(160/64) = 9
  replica 1: start = 5, moved = 240, finish = 5 + ceil(240/32) = 13
  -> replica 0, ttft = 9; resident[0] = 360

Request 2: kv = (100+100)*4 = 800
  replica 0: 360 + 800 = 1160 > 1000, admission fails
  replica 1: 800 <= 1000, start = 15, moved = 1200, finish = 15 + ceil(1200/32) = 53
  -> replica 1, ttft = 53; resident[1] = 800

Request 3: kv = (10+5)*4 = 60
  replica 0: 420 <= 1000, start = max(16, 9) = 16, finish = 16 + ceil(60/64) = 17
  replica 1: 860 <= 1000, start = max(16, 53) = 53, moved = 90, finish = 53 + ceil(90/32) = 56
  -> replica 0, ttft = 17
```

```
placement = [0, 0, 1, 0]
ttft      = [6, 9, 53, 17]
wire      = [420, 1200]
resident  = [420, 800]
rejected  = 0
makespan  = 53
```

## Constraints

- `prompt_lens[i] >= 1`, `gen_lens[i] >= 0`, `kv_bytes_per_token >= 1`, `prefill_tps >= 1`;
- `caps[d] >= 0`, `bws[d] >= 1`, `tps[d] >= 1`;
- `caps`, `bws`, and `tps` must have the same length, as must `prompt_lens` and `gen_lens`; otherwise raise immediately;
- the request list may be empty, in which case the ledgers are empty and `makespan` is 0.
