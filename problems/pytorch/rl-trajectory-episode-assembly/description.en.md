# RL Systems: Multi-Segment Episode Assembly and Truncation Bootstrap

## Problem

What the rollout side hands to the trainer is a **multi-segment mixed** record stream: one buffer tensor concatenates several episodes (each trajectory of a multi-turn conversation is one segment); the critic logs \(V(s_t)\) at every step, and at a segment's end an extra value-head pass over the final observation yields \(V(s_{end})\). But the downstream advantage/return computation (see `to-gae`) wants per-episode aligned \((a_t, r_t, V(s_t), done_t, next\_value_t)\) slices. This **assembly** layer has two classic bugs:

1. Treating a **truncation** (step limit / max-generation-length cutoff) as a true termination and zeroing the bootstrap — value regression becomes systematically biased and the model learns that "surviving to the limit is a good ending". Per the Gymnasium `TimeLimit.truncated` convention, a truncated segment's next_value must be the terminal observation's \(V(s_{end})\); **only a true termination (terminated) gets 0**.
2. Naively taking the successor value with a `roll(-1)` along the concatenated array — the last step of one episode picks up the **first step's value of the next episode**, leaking value across segments. The role of `done` is precisely to forbid "take the next record" at boundaries and route to the bootstrap/0 branch instead.

This problem is assembly and alignment only — **the GAE recursion itself is not re-derived here** (that belongs to `to-gae`).

## Rules

Seven 1-D CPU inputs: `episode_id` (int64 \((T,)\); one episode forms exactly one maximal contiguous segment; ids are globally unique and may arrive out of order), `terminated` / `truncated` (int64 \((T,)\) 0/1 flags), `actions` (int64 \((T,)\), pass-through), `rewards` / `values` (float32 \((T,)\), i.e. \(r_t\) and \(V(s_t)\)), and `final_value` (float32 \((K,)\), where \(K\) is the number of segments, indexed by **segment arrival order**: the terminal-observation value \(V(s_{end})\) of each episode; the entry of a truly terminated segment exists but is unused). Guaranteed input: each segment carries exactly one flag set, at its last step only; all other steps have both flags 0.

Assembly contract:

1. **Training order**: segments are re-ordered by ascending `episode_id`; steps keep their chronological order inside a segment.
2. **done**: 1 at each segment's last step, 0 elsewhere; a truncated ending also counts as done.
3. **next_value three-way rule** (aligned to training order): non-last step within a segment → the `value` of the next step in the same segment; segment-last with `truncated=1` → that segment's `final_value` entry (bootstrap restored); segment-last with `terminated=1` → \(0\). No position may ever take a record belonging to a different segment as its successor.

Return a 7-tuple (all aligned to training order): `actions`, `rewards`, `values`, `dones`, `next_values` (each \((T,)\)), `lengths` (\((K,)\), per-segment step counts in training order) and `starts` (\((K,)\), each segment's start index in training order, an exclusive prefix sum). For \(T=0\) return seven 0-element tensors. Ready-made RL libraries are forbidden.

## Example

Segments arrive with ids 7, 3, 8 (out of order); segment 0 truly terminates, segments 1 and 2 are truncated:

```
episode_id = [7, 7, 7, 3, 3, 8, 8, 8]
terminated = [0, 0, 1, 0, 0, 0, 0, 0]
truncated  = [0, 0, 0, 0, 1, 0, 0, 1]
actions    = [10, 11, 12, 20, 21, 30, 31, 32]
rewards    = [1.0, 0.0, 0.5, 0.5, 2.0, 1.0, 1.0, 0.0]
values     = [0.5, 0.625, 0.75, 0.25, 0.375, 0.125, 0.25, 0.3125]
final_value= [0.9375, 0.875, 0.625]        # indexed by segment arrival; segment 0 terminates, 0.9375 is unused
```

Re-sorted into training order by id (3, 7, 8), the output is:

```
actions     = [20, 21, 10, 11, 12, 30, 31, 32]
rewards     = [0.5, 2.0, 1.0, 0.0, 0.5, 1.0, 1.0, 0.0]
values      = [0.25, 0.375, 0.5, 0.625, 0.75, 0.125, 0.25, 0.3125]
dones       = [0, 1, 0, 0, 1, 0, 0, 1]
next_values = [0.375, 0.875, 0.625, 0.75, 0.0, 0.25, 0.3125, 0.625]
lengths     = [2, 3, 3]
starts      = [0, 2, 5]
```

Reading `next_values` position by position: position 1 is the truncated last step of the id=3 segment, bootstrapped with `final_value[1]=0.875`; position 4 is the truly terminated last step of the id=7 segment, taking \(0\); position 7 is the truncated last step of the id=8 segment, taking `final_value[2]=0.625`. A naive `roll(-1)` would give the id=7 last step 0.25 (the first value of the id=3 segment — cross-segment leak) and the id=3 last step 0.125 (the first value of the id=8 segment) — neither matches the correct 0.875 / 0.

## Constraints

- CPU; \(T \le 4096\); \(K \ge 0\); int64 outputs checked with `torch.equal`, float32 with `allclose(atol=1e-5, rtol=1e-5)`.
- Invariants: \(\sum\) `lengths` \(=T\); `starts[0]=0` and consecutive `starts` differ by exactly the corresponding `lengths`; `dones` sums to \(K\); at every done position the `next_value` is either that segment's `final_value` (truncated) or exactly \(0\) (terminated) — never a value from a neighbouring segment; at every non-done position the `next_value` equals the `values` entry of the next training-order position (same segment).
- Edges: single-step segments (first = last) are legal; all-truncated and all-terminated batches are legal; \(T=0\) follows the empty convention; when ids already arrive ascending, training order = arrival order.
- Division of labour: this problem is **multi-segment assembly only** — segmentation, re-ordering, and the done/bootstrap three-way rule. The GAE backward recursion \(\hat A_t=\delta_t+\gamma\lambda\hat A_{t+1}\) and its degenerate-parameter probes live in `to-gae` (a single segment with an implicit terminal \(V_T=0\), i.e. the "all true termination" special case); cross-rank re-sharding and micro-batch packing live in `rl-rollout-shard-repack`. The per-segment \((rewards, values, next\_values, dones)\) produced here is exactly what a downstream per-segment GAE call consumes.
