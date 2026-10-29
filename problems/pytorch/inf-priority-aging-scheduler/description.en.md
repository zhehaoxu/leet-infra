# Priority Admission Scheduling with Aging

## Problem

Inference services usually hand out slots by priority: VIP requests jump the queue, batch jobs wait at the back. Pure priority ordering has a classic failure mode — **starvation**: as long as high-priority requests keep arriving, a low-priority request never gets a slot.

The standard fix is **aging**: every request that waits long enough gains one level of effective priority, so it creeps toward the head and is eventually served. This problem simulates that rule in integer steps: given arrival times, service durations, base priorities, a slot limit, an aging interval and a priority cap, output each request's admission and finish step plus the longest wait.

## Rules

Time is measured in integer steps; step 0 is the first time point.

Each step:

1. **Admission**: free slots = `max_batch − number of running requests`. Among all requests that have arrived and are not yet admitted, repeatedly pick the one with the highest **effective priority at the current time `t`**:

   ```
   eff = min(priority_cap, priority[i] + (t − arrivals[i]) // aging_interval)
   ```

   Ties break in order: higher base priority → earlier arrival → smaller index. A request admitted in this step starts running (and holding a slot) immediately.
2. **Run**: every running request consumes one service step; a request whose service ends here completes in this step (its finish step is this step), and the slot it releases becomes usable from the **next** step.
3. **Idle**: if nothing is running and no request has arrived yet, jump the clock to the next arrival step; idle steps produce no accounting.

Return a triple:

- `admit`: each request's admission step (absolute);
- `finish`: each request's finish step, always `admit[i] + service[i] − 1`;
- `max_wait`: the largest `admission step − arrival step` over all requests.

## Examples

**Example 1** (aging lets a low-priority request overtake): `arrivals = [0, 0, 1, 2, 3, 4, 5, 6]`, `service = [1] * 8`, `priority = [5, 0, 5, 5, 5, 5, 5, 5]`, `max_batch = 1`, `aging_interval = 1`, `priority_cap = 9`

```
t=0  r0 (eff 5) vs r1 (eff 0)              -> r0
t=1  r1 (eff 1) vs r2 (eff 5)              -> r2
t=2  r1 (2), r3 (5)                        -> r3
t=3  r1 (3), r4 (5)                        -> r4
t=4  r1 (4), r5 (5)                        -> r5
t=5  r1 (5), r6 (5): tie, base 5 > 0       -> r6
t=6  r1 (6), r7 (5): aged one level higher -> r1   <- the low-priority request is rescued here
t=7  r7 (5)                                -> r7
```

```
admit    = [0, 6, 1, 2, 3, 4, 5, 7]
finish   = [0, 6, 1, 2, 3, 4, 5, 7]
max_wait = 6                # r1 escapes at step 6; without aging it would wait all 7 steps
```

**Example 2** (no headroom for aging): same stream, but `priority_cap = 5` (equal to the highest base priority)

```
t=5  r1 (5), r6 (5): tie, the higher base priority r6 wins
t=6  r1 (5), r7 (5): same, r7 wins
t=7  r1 (5)                                -> r1
```

```
admit    = [0, 7, 1, 2, 3, 4, 5, 6]
finish   = [0, 7, 1, 2, 3, 4, 5, 6]
max_wait = 7                # with the cap in place aging can never pass 5; only the end of the stream helps
```

**Example 3** (two slots, and a freed slot is usable next step): `arrivals = [0, 0, 0]`, `service = [4, 2, 2]`, `priority = [1, 1, 0]`, `max_batch = 2`, `aging_interval = 2`, `priority_cap = 3`

```
t=0  admit r0, r1 (both eff 1, tie broken by index; r2 is only 0)
t=1  r0 has 3 steps left, r1 has 1; no free slot
t=2  r1 finished at t=1, so a slot is free: r2 eff = 0 + 2//2 = 1 -> admitted
t=3  r0 and r2 finish together
```

```
admit    = [0, 0, 2]
finish   = [3, 1, 3]
max_wait = 2
```

## Constraints

- At most 64 requests; `arrivals[i] >= 0` (repeats and out-of-order allowed); `service[i] >= 1`; `priority[i] >= 0`.
- `max_batch >= 1`, `aging_interval >= 1`, `priority_cap >= max(priority)`.
- The effective priority is recomputed at **every pick** using the current time; it is not frozen once per step.
- The two lists and the integer are Python `int`s; only the numbers are judged, any scheduling loop is fine.
