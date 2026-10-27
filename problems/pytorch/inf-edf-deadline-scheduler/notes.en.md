# Solution Notes · EDF with Wait Aging

## Three fields in the sort key

The per-step selection is one key:

```
key(r) = (0 if wait[r] >= max_wait else 1, deadline[r], r)
```

Group by "is it aged" first (0 sorts ahead), then by deadline, then by id to break ties. Writing it as a single tuple is much harder to get wrong than "sort twice and concatenate".

## Where `wait` gets updated is the usual bug

`wait` means "consecutive skipped steps", so it changes in exactly two places:

- chosen → reset to 0 (even if the selection came *from* aging, otherwise it stays at the front next step);
- ready but not chosen → increment.

**A request that has not arrived must not increment**, otherwise `wait` becomes idle time rather than waiting time, and a late arrival walks in with a huge `wait` and jumps the queue.

## Aging is not "promote the single hungriest one"

Promoting only the hungriest while leaving other aged requests behind degenerates the aged subset back to EDF, and you get churn: just-promoted requests get pushed out again. Promote the whole set of requests that satisfy the threshold as one block, still EDF-ordered inside. With `max_batch >= 1` an aged request then waits at most `aged_group_size / max_batch` more steps.

## `missed` and "never finished" are the same thing

`finish == -1` means `max_steps` ran out before the request completed, which is a miss in every practical sense and must be counted. Comparing only `finish > deadline` silently drops it (-1 is not greater than any non-negative deadline) and the statistic looks optimistic.

## A cheap self-check

Set `max_wait` larger than `max_steps`: `promotions` must be 0 and the output must match pure EDF exactly. That is the easiest probe for "did the aging logic accidentally touch the normal path".

## Complexity

One sort per step, O(N log N), so O(max_steps × N log N) overall. Real machines run hundreds of requests over tens of thousands of steps, so EDF keeps the per-step constant small with heaps or timing wheels instead of re-sorting.

## Related problems

- `inf-continuous-batching-scheduler`: the FCFS version with no deadlines.
- `inf-kv-block-admission`: the same question along the KV block axis.
- `inf-cache-aware-routing`: move the decision upstream so a single replica queues less.
