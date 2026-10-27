# searchsorted Bucketing and bincount Histogram

## Problem

"Binning continuous values, then counting each bin" is a paired operation: first look up a bucket per element (`searchsorted`), then count the bucket indices (`bincount`). It is the shared low-level primitive behind inverse-CDF sampling (given a uniform draw \(u\), find the first position on the cumulative distribution that is \(\ge u\) — see `top-p-sampling`, `rl-per-priority-replay`), MoE expert-load counting (`moe-switch-aux-loss`) and nearest-codebook encoding (`quant-nf4-quantile-grid`'s 16-segment quantile codebook).

This task chains the two steps into one function, and all the difficulty lives in two things: **which side of the boundary is closed** and **the bin count is fixed**:

- with `right=False` the bucket index is "how many boundaries are strictly less than \(x\)", i.e. the smallest \(i\) with \(\text{boundaries}[i] \ge x\);
- with `right=True` it is "how many boundaries are \(\le x\)", i.e. the smallest \(i\) with \(\text{boundaries}[i] > x\).

The two only become distinguishable when \(x\) is **exactly equal** to a boundary, or when boundaries repeat — the worked example deliberately contains both. Repeated boundaries create bins that are always empty, yet `counts` **must** have length \(M+1\): an empty bin still occupies a slot, which is the most substantive difference from a bare `torch.bincount` (which truncates trailing empty bins).

CPU, fully deterministic, no RNG calls.

## Rules

- The full signature, parameter meanings and defaults live only in the starter docstring; that is the sole judging contract. `boundaries` is a 1-D tensor already sorted ascending (duplicates allowed, \(M=0\) allowed), `x` has any shape and the same dtype as `boundaries` (int64 or float32/float64), `right` defaults to False, `weights` defaults to None.
- **Bucket lookup**: return an int64 tensor `idx` with the same shape as `x`, each element \(i\in[0,M]\) defined by the two counting rules above. Equivalent interval reading: with `right=False` bin \(i\) collects \((\text{boundaries}[i-1],\ \text{boundaries}[i]]\) (right-closed), with `right=True` it collects \([\text{boundaries}[i-1],\ \text{boundaries}[i])\) (left-closed). The "left-closed, right-open" sentence in the starter docstring is the `right=True` reading — under `right=False` the closure flips, and the two unbounded outer bins flip with it (\((-\infty,\text{boundaries}[0]]\) and \((\text{boundaries}[M-1],+\infty)\)). With duplicate boundaries some inner bins are identically empty; the counting definition of `idx` always wins.
- **Counting**: `counts` is a 1-D tensor of length **exactly \(M+1\)**, where `counts[i]` is the number of elements of `idx` equal to \(i\) (no `weights`) or the sum of their weights (`weights` given). `idx` is flattened before counting, so a multi-dimensional `x` only affects the shape of `idx`.
- **dtypes**: `idx` is always int64; without weights `counts` is always int64; with weights `counts` is floating point and its dtype follows `weights` (float32 → float32, float64 → float64). Judging only feeds floating-point `weights`. Note in passing that native `torch.bincount` silently promotes integer `weights` to float64, which is *not* the "dtype follows weights" contract here — don't copy that behaviour.
- **Identities** (used as probes): unweighted `counts.sum() == x.numel()`; weighted `counts.sum() == weights.sum()`.
- **Empty inputs**: with an empty `x`, `idx` is empty and `counts` is all-zero with length \(M+1\); with empty `boundaries` (\(M=0\)) every element lands in the single bin 0, so `idx` is all zeros and `counts` has length 1. Neither case raises.
- **Error contract**: `boundaries` not 1-D → RuntimeError; `weights` shape mismatching `x` → ValueError. `boundaries` are guaranteed sorted; no need to handle unsorted input.

## Example

`boundaries = [1.0, 3.0, 3.0, 7.0]` (\(M=4\), so `counts` has length 5), `x = [0.0, 3.0, 5.0, 9.0]`:

**right=False** (default):

| x | boundaries strictly less | idx |
|---|---|---|
| 0.0 | none | 0 |
| 3.0 | `[1]` | 1 |
| 5.0 | `[1,3,3]` | 3 |
| 9.0 | all four | 4 |

`idx = [0, 1, 3, 4]`, `counts = [1, 1, 0, 1, 1]` (bin 2's interval degenerates to the empty set \((3,3]\) under `right=False`, so it is identically 0).

**right=True**: for `x=3.0` three boundaries (\([1,3,3]\)) are \(\le x\), the other elements are unchanged, giving `idx = [0, 3, 3, 4]` and `counts = [1, 0, 0, 2, 1]`.

The branches differ on exactly one element (`x=3.0`; `0.0`, `5.0`, `9.0` sit far from the repeated run), which moves from bin 1 (\((1,3]\)) to bin 3 (\([3,7)\)). The bin that is *identically* empty because of the repeated run \([3,3]\) is index 2 under both settings — `counts[2] == 0` either way — and `counts` keeps length 5 in both. That is the minimal discriminator for this task's two core rules.

Weighted example: `boundaries=[1,3,7]`, `x=[0,2,2,5,9]`, `weights=[1,2,3,4,5]`. Unweighted: `idx=[0,1,1,2,3]`, `counts=[1,2,1,1]`. Weighted: `counts=[1.0, 5.0, 4.0, 5.0]` (bin 1 collects \(2{+}3\), bin 2 only collects 4).

## Constraints

- Return the tuple `(idx, counts)`: `idx` has the same shape as `x` and is int64; `counts` is 1-D with length \(M+1\), int64 or the floating dtype of `weights`.
- Judging uses `torch.equal` (indices and integer counts) and \(10^{-6}\) (floating counts); the implementation path is free — binary search, broadcasting `boundaries` against `x` and summing comparisons, or accumulating with `index_add_`.
- No random numbers; untouched bins must emit an explicit 0 rather than be omitted.
