# Explained: diff (n-th Order Difference with prepend/append)

## Origin

`torch.diff` has existed since 1.5 and mirrors `np.diff` (the `prepend`/`append` arguments are exactly numpy's `np.diff(..., prepend=, append=)` interface). Its uses are everywhere: turning a position-id sequence into step sizes (checking whether `position_ids` stay consecutive in RoPE extrapolation), converting token spans into boundaries, detrending time series, finite-difference gradient checking (the same family as the gradcheck in item 65, which there compares $f(x+\epsilon)-f(x-\epsilon)$ against the analytic gradient), and criteria such as "a negative second difference means non-convex".

## The Difference Operator and Its Two Subtleties

The first difference along axis $d$ is a $(L-1)\times L$ banded matrix acting on that axis:

$$(\Delta x)_i = x_{i+1} - x_i, \qquad \Delta^{(n)} = \underbrace{\Delta^{(1)} \circ \cdots \circ \Delta^{(1)}}_{n}$$

Implementation-wise there are only two traps:

1. **Any axis.** The slicing form `x[:, 1:] - x[:, :-1]` hard-codes the axis name, so switching to `dim=1` means rewriting it. `torch.narrow` abstracts "take $l$ elements starting at $s$ along axis $i$" into a generic op, so

   ```
   y = y.narrow(axis, 1, size - 1) - y.narrow(axis, 0, size - 1)
   ```

   works for every axis, with `size` re-read each round (the length shrinks by 1 per pass).
2. **Concatenate the boundaries first.** The **last row** of `prepend` is adjacent to the first row of `x`, and the first row of `append` follows the last row of `x`, so the order must be `cat([prepend, x, append], dim=axis)`; the output length is

   $$L_{\text{out}} = \dim_x + \dim_p + \dim_a - n$$

   counting None sides as 0. An implementation that subtracts `n` only once will over-report the length as soon as $n\ge 2$.

The $n$-th difference equals "apply the first difference $n$ times in a row", and the judge verifies that recursive definition directly with the identity `diff(x, n=2) == diff(diff(x), n=1)`.

## Relation to the Cumulative Sum

The difference is a left inverse of `cumsum`: $\Delta\big(\mathrm{cumsum}(x)\big)_i = x_{i+1}$ (the first element is dropped) and $\mathrm{cumsum}(\Delta x) = x - x_0$. Reconstructing the original sequence from differences therefore needs an extra initial value or a `prepend` — which is exactly why numpy and torch keep those two parameters: without a boundary value, differencing is not invertible.

## Core Points and Pitfalls

1. **`n=0` must return a copy**: the judge asserts `torch.allclose(solve(x, n=0), x)`, and semantically the input object should not be handed back (mutating the output would then hit the input).
2. **A non-positive resulting length raises**: with an axis length of 2 and `n=3` torch returns an empty tensor, while this problem requires ValueError — so each round must check `size < 2` before calling `narrow`.
3. **Granularity of the `prepend`/`append` check**: same ndim, agreement on every dimension except `dim`, same dtype; comparing only the number of shape entries misses "dimension 1 disagrees".
4. **bool is not int**: `n=True` would slip into the `n==1` branch, so the reference excludes it explicitly.
5. **Differencing never changes dtype**: plain float32 suffices (no stabilisation needed), unlike the other members of this batch.

## How It Differs from Neighbouring Problems

- `reduce-axis` / `tensor-shape-indexing`: reductions and shape manipulations, with no notion of "subtract two neighbours".
- `concat-stack-split`: concatenation is only the means by which this problem handles `prepend`/`append`; that one does not examine differencing.
- `repeat_interleave` (order364): the sister problem in the same batch about reconstructing an index mapping — there one builds an index vector and calls `index_select`, here one builds views and subtracts with `narrow`.

## Cross-Checking and Complexity

- The oracle is `torch.diff`, over six `n` settings, six positive/negative `dim` spellings, and single/double boundary combinations; plus length-formula checks, "second difference = difference of the first", the constant-1 / all-zero pattern for a linear sequence, element-by-element verification on a middle axis, and eight illegal-argument probes.
- Complexity $O(nN)$ for $N$ elements, one elementwise subtraction per round; no float64 needed, so judging runs from microseconds to milliseconds.
