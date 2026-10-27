# LpPool and maxpool Indices/unpool

## Problem

Three kinds of pooling share one window geometry (`floor((n + 2p − k)/s) + 1`), but each hides a different trap:

- **LpPool**: the output is the \(p\)-norm of \(|x|\) inside the window, \(\big(\sum_i |x_i|^p\big)^{1/p}\) — not a mean, and not signed. `p=1` is a plain sum of \(|x|\) (`avg_pool2d` differs by a factor \(1/k^2\), and the `p=1` branch of native `F.lp_pool2d` is a **signed** sum); only `p=∞` becomes maxpool. The task restricts \(p\) to finite values \(\ge 1\) and takes \(|x|\) for **every element** — this is the deterministic convention defined by this repository, and judging probes it with negative inputs.
- **maxpool + indices**: besides the maximum, return "which source position the maximum came from". The index is the row-major flattened offset over the **unpadded** input spatial map (\(\text{row}\times W + \text{col}\)), and ties take the row-major-first one. Padding must be filled with \(-\infty\), not 0, otherwise a negative region would be topped by the pad value.
- **maxunpool**: use those indices to scatter each window maximum back to its source position, zeros everywhere else — the sparse inverse of maxpool. It is only well-defined when windows do not overlap, so the task requires **stride ≥ kernel in every dimension** and rejects overlapping configurations outright.

One function, four modes, CPU, fully deterministic, no RNG calls.

## Rules

- The full signature, parameter meanings and defaults live only in the starter docstring; that is the sole judging contract. `mode` is one of `"lppool2d" | "lppool1d" | "maxpool_idx" | "maxunpool"`, anything else raises ValueError; `kernel_size/stride/padding` accept an int or a length-1/2 integer sequence, and `stride=None` defaults to the kernel per dimension.
- **Window geometry (same for all three spatial modes)**: output side \(o = \lfloor (n + 2p_{\text{pad}} - k)/s \rfloor + 1\), which must satisfy \(o\ge 1\); `padding` on each side may not exceed half of the corresponding kernel dimension (native `F.max_pool2d` raises RuntimeError here, this task unifies it to ValueError).
- **lppool** (1-D/2-D): output \(= \big(\sum_{\text{window}} |x_i|^p\big)^{1/p}\), zero padding, `p` must be \(\ge 1\) and finite; the output is \((N,C,ol)\) for 1-D or \((N,C,oh,ow)\) for 2-D, with the same dtype as `x`. `lppool1d` requires `x` to be exactly 3-D with kernel/stride/padding equal across dimensions.
- **maxpool_idx**: returns the tuple `(out, indices)`. `out` has `F.max_pool2d` semantics; `indices` is int64 with the same shape as `out` and holds the row-major flattened offset of the maximum within the **original** (unpadded) \(H\times W\) map. This mode ignores `p` entirely.
- **maxunpool**: input is the small map \((N,C,h,w)\) plus `indices` (same shape as `x`, integer dtype); output is \((N,C,H,W)\): each window value is written to the position its index points at, everything else is 0. With `output_size=None` the size is inferred as \(H=(h-1)s_h+k_h\) per dimension; an explicit value must be \(\ge\) the small-map size in every dimension. Requires `padding == 0` and `stride >= kernel` per dimension.
- **Rank constraints**: `lppool1d` needs 3-D, the other three need 4-D; otherwise ValueError.
- **Error contract**: every invalid argument here (bad mode, bad rank, bad `p`, padding over the limit, window that does not fit, `indices` shape/dtype/range, too-small `output_size`, overlapping configuration) raises **ValueError** — deliberately different from native's RuntimeError, and judging asserts ValueError.
- Returns new tensors and never mutates its inputs; ties in `maxpool_idx` take the row-major-first maximum.

## Example

\(x\) is (1,1,3,3):

\[x=\begin{bmatrix}1&2&3\\4&5&6\\7&8&9\end{bmatrix},\qquad k=2,\ s=2,\ p=2\]

The output side is \(o=\lfloor(3-2)/2\rfloor+1=1\) (row 3 and column 3 are dropped by the floor and belong to no window); the single window is \([1,2,4,5]\):

- `lppool2d`: \(\sqrt{1^2+2^2+4^2+5^2}=\sqrt{46}\), measured **6.7823300362** in float32 (the exact value is \(6.7823299831\ldots\), well inside the \(10^{-6}\) tolerance).
- `maxpool_idx`: `out = 5.0`; the maximum 5 sits at input coordinate \((1,1)\), whose row-major flattened offset is \(1\times 3 + 1 = \) **4**, so `indices = 4`.
- `maxunpool` (with `output_size=(3,3)` and the index above): a \(3\times3\) tensor of zeros except position \((1,1)\), which holds 5.

That last step **must** pass `output_size=(3,3)` explicitly: `output_size=None` infers \(H=(1-1)\times2+2=2\), i.e. a \(2\times2\) canvas, and then the index 4 is out of range and raises ValueError — when the leftover row/column was discarded by the floor, the inference formula yields the *smallest possible* output, not the original input size.

## Constraints

- Shapes: `x` is \((N,C,H,W)\) for `lppool2d`/`maxpool_idx` and \((N,C,L)\) for `lppool1d`; for `maxunpool` `x` is the small map and `indices` has its shape.
- dtypes: `lppool*` output matches `x` (float32/float64); `maxpool_idx` returns int64 `indices`; `maxunpool` output matches `x`'s dtype.
- Judging tolerance: \(10^{-6}\) with `torch.allclose` for float32 (measured maximum deviation against native `F.lp_pool2d`/`F.max_pool2d` is of order \(10^{-7}\)), \(10^{-12}\) for float64; `indices` is compared exactly with `torch.equal`.
- No random numbers; an all-zero window (including windows made of zero padding) must output 0, not NaN or anything undefined.
