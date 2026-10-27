# SLO-Aware EDF Scheduling with Wait Aging

## Problem

The SLO attached to an inference request is usually a **deadline step** — for the first token or for the whole request. Each round the scheduler can only advance a limited number of requests by one step, so who runs is the core decision. Earliest-deadline-first (EDF) minimises total misses, but it is strictly absolute: as long as tighter deadlines keep arriving, a long request with a loose deadline may never get a turn. That is starvation.

The standard patch is **aging**: a request skipped too many times in a row is promoted to the front. This problem stacks the two and asks for per-request start/finish steps, the miss count, and how many times aging actually fired.

## Task

Implement `solve(arrival, num_tokens, deadline, max_batch, max_wait, max_steps)`:

- Inputs: `arrival` `(N,)` arrival step; `num_tokens` `(N,)` decode steps needed; `deadline` `(N,)` must finish at or before this step; `max_batch` how many run concurrently; `max_wait` consecutive skips that trigger promotion; `max_steps` simulation limit.
- Each step:
  1. ready = arrived and unfinished;
  2. every request with `wait >= max_wait` is moved to the front as a block (aging);
  3. both groups are sorted by `(deadline asc, id asc)`;
  4. the first `max_batch` each run one step (`num_tokens -= 1`), recording the finish step when it hits 0;
  5. `wait` is reset to 0 for the chosen ones and incremented for the ready-but-unchosen ones.
- Outputs: `start` `(N,)` first run step (-1 = never ran), `finish` `(N,)` (-1 = unfinished), `missed` int64 scalar (finished after its deadline, or never finished), `promotions` int64 scalar (total number of aging-driven selections).

## Example

```
arrival = [0, 0], num_tokens = [3, 1], deadline = [100, 100]
max_batch = 1, max_wait = 2, max_steps = 6

step 0  wait = [0, 0], nothing is aged; equal deadlines -> smaller id -> run r0. r1's wait becomes 1
step 1  wait = [0, 1], nothing is aged -> run r0 again. r1's wait becomes 2
step 2  r1's wait = 2 >= max_wait -> aged to the front -> run r1, r1 finishes   promotions = 1
step 3  r0 runs its last step -> finished

start      = [0, 2]
finish     = [3, 2]
missed     = 0
promotions = 1
```

## Constraints

- **Aging is a block promotion, not an insertion**: every request with `wait >= max_wait` goes ahead of the rest, still ordered among themselves by `(deadline, id)`.
- **The threshold is `>=`**: `max_wait = 2` means "already skipped twice", not "skipped three times".
- **`wait` grows only for "ready but not chosen"**: requests that have not arrived do not accumulate, and chosen requests reset to 0.
- **`promotions` counts selections**: three aged requests chosen in one step contributes 3.
- **`missed` accounting**: `finish > deadline` or `finish == -1` both count, including "did not finish because `max_steps` ran out".
- **`max_batch >= 1`**: at least one request must be able to run each step, otherwise progress cannot be guaranteed.
