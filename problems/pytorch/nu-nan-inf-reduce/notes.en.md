# Notes: Three Ways to Implement Skipping Reductions and the Trap in Each

## Origin

`torch.nanmean`/`nanmax` are public APIs, but the hand-written version shows up in concrete engineering spots: an all-reduce fallback for a sparse multi-device gather (ranks that never flushed are filled with NaN), the attention denominator under causal/document-boundary masks (masked positions must not enter the softmax denominator nor turn the whole row into NaN), and "mean over valid samples" in data-cleaning scripts. Interviews use it for two true/false questions: what exactly is the `nanmean` denominator, and who wins when NaN and inf meet.

## Three Routes

1. **Sentinel replacement**: swap NaN for the identity element of the reduction (`sum` -> 0, `max` -> -inf, `min` -> +inf), then call the native reduction. Shortest route, but `mean` still needs `x.isnan().sum(dim)` for the true denominator, and an all-NaN segment leaks its sentinel — `max` would return -inf instead of NaN — so a second "valid count == 0" mask has to rewrite the result to NaN.
2. **Mask first, then `where`**: `valid = ~x.isnan()`, values via `torch.where(valid, x, fill)`, denominator via `valid.sum(dim)`. Equivalent to route 1 but keeps the mask explicit, which makes `keepdim` and the 0-D output easier to line up shape-wise.
3. **Pure-Python segments**: `movedim` the reduced axis to the end, `reshape(-1, n)` into segments, and reduce each with the builtin `sum/max/min`. The reference solution uses tensor code while the oracle uses this route, so the two are structurally unrelated — which is why the judge catches both "segment length as denominator" and "inf treated like NaN".

## Core Ideas and Pitfalls

- **`amax` has no `initial=`**: in the torch version pinned by this repository `Tensor.amax` does not accept the `initial` keyword, so the "-inf to start" trick dies with a TypeError; the two-argument `torch.max` behaves differently again. The portable answer is to do the sentinel replacement and the mask rewrite yourself.
- **`amax` propagates NaN**: `torch.tensor([nan, 1.0]).amax()` returns NaN (unlike the builtin Python `max`, whose result depends on comparison order, tensor reductions have a fixed propagation rule). So "amax first, then extract the NaN positions" is far more reliable than "hope amax ignores NaN".
- **A reduced axis of length 0**: `amax` raises RuntimeError on an empty segment, so a `reduced_len == 0` branch is mandatory — this problem specifies 0.0 for `sum` and NaN otherwise, i.e. an empty segment behaves like count == 0.
- **Subtracting inf**: an assertion written as `abs(a - b) <= tol` compares inf against inf as `inf - inf = nan` and therefore never passes. A short-circuit `a == b` before the subtraction is required; this is a shared trap in every numeric problem that admits inf.
- **NaN equality semantics**: `nan == nan` is false in a Python list, so element-wise assertions must be NaN-aware (both sides NaN counts as equal) instead of `assert got.tolist() == want`.
- **The `mean` denominator**: the nanmean of `[nan, 1.0, nan]` is 1.0, while a segment length of 3 gives 0.333. This is a standalone assertion in the tests.

## How It Differs from Neighboring Problems

`reduce-axis` only tests axis semantics (which axes disappear, keepdim shapes, broadcasting back) and has no NaN/inf branch at all; `flash-attention-tile` and `nu-online-softmax-rescale` mask with -inf, so they only care about the running max and an exp sum and never need "skip" semantics. The exclusive point here is defining the full cross product of NaN, ±inf and empty segments across four modes.

## Cross-Check Basis and Complexity

The oracle is the pure-Python filter-then-reduce over segments. Well-conditioned random data (35% NaN injected, shapes of 1/2/3 dimensions, every legal dim plus None, both keepdim settings) must match within 1e-12 relative tolerance with NaN positions aligned bitwise; the example must match bitwise. Complexity is one \(O(N)\) pass plus a bool mask (\(N\) bits) and a count tensor.
