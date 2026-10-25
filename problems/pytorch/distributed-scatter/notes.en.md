# Approach

## Interviewer's Intent

Scatter / All-Gather are a pair of inverse operations and are often tested together. If you cannot distinguish "slicing a batch of data and distributing it to each GPU" from "collecting each GPU's shard back", you will write them in the wrong direction in data-parallel loading or tensor-parallel layouts. This problem is the mirror image of All-Gather; the simpler the implementation, the more you must watch the semantic details.

## Semantics

The source rank holds one-dimensional complete data `full` (length $L$, divisible by $W$). Scatter **splits it evenly into $W$ segments**; the `i`-th segment `full[i*chunk:(i+1)*chunk]` goes only to rank `i` (`chunk = L/W`). The output is a list in which **each rank gets a different segment**.

## Comparison with Neighboring Primitives

- **All-Gather**: each rank has a shard → everyone receives the concatenated whole (the inverse of Scatter).
- **Scatter**: one complete copy → one shard per rank (**no summation**).
- **Reduce-Scatter**: each rank has one complete copy → sum first, then each takes one segment.
  The easiest pair to confuse is Scatter vs Reduce-Scatter: the former has only one copy of input data and performs no cross-rank reduction.

## Implementation Notes

After `full.reshape(-1)`, slice with a stride of `chunk`; use `.clone()` so that each segment is independent and does not share `full`'s storage, satisfying "do not modify the input". The test `check_inverse_of_allgather` pins down the inverse invariant **`cat(scatter(full)) == full`**.

## Real-world Usage

In data parallelism, `DistributedSampler` splits and distributes the whole dataset by rank, which in essence is the idea of a scatter: each worker takes only its own $\frac1W$ of the samples. When initializing tensor parallelism, scatter is also often used to slice a large weight along a dimension into one piece per GPU.

## Grading Approach

`test_cases.py` verifies: correct slicing order, `cat` reassembling back to the original data, the degenerate cases `chunk=1` and `world=1`, cross-checking segment by segment over multiple $(L, W)$ combinations, and no modification of the input.
