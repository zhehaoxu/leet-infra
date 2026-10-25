# Solution Notes · PyramidKV

## Budget Conservation

The design goal is: the sum of kept counts over all layers ≈ the total when every layer uses the same budget (so that compression_ratio takes effect).

$(\text{max} + \text{min}) \cdot \text{num\_layers} / 2$ is the arithmetic-series sum; setting it equal to $\text{max\_cp} - W$ times the number of layers (dropping the window part) yields `max_cp = W + q_len*(1-cr)`.

## The Cap Logic

When `max_num` is large and close to q_len, `min_num` gets pushed below `window_size` — in that case clip `max_num` to `q_len - window_size` and recompute `min_num` (keeping the total sum unchanged).

## Fallback Condition

If, even after clipping, `q_len >= max >= min >= window` still fails, q_len is too short — the pyramid cannot hold, so it degrades back to SnapKV with a uniform per-layer budget.

## round vs int

kvpress uses `round(x)` rather than `int(x)` — Python's `round` is banker's rounding, which differs subtly from the intuitive half-up rounding (0.5 is handled specially). This problem keeps using `round`.

## Relation to LUKV / PerLayer

PyramidKV merely formalizes "how many to keep per layer" into a formula. `LUKVPress` provides an **arbitrary budget curve**, and `PerLayerCompressionPress` allows a **different compression_ratio per layer** — all belong to the same family of "per-layer budget allocators", orthogonal to "how to score" (SnapKV / KNorm).
