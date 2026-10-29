# Safe p-Norm: Scale by the Segment Maximum Before Reducing

## Problem Description

"Sum of squares then square root" is the line least likely to be written wrong, yet it fails silently in two directions: squaring large numbers overflows to \(+\infty\) and squaring tiny ones underflows to \(0\), so the same function returns badly wrong answers at both ends without making a sound. The production-correct approach is to first scale a whole segment into \([0,1]\) by its largest magnitude, reduce, then multiply the scale back. This problem asks for that scaled p-norm, supporting any \(p>0\) (including \(p=+\infty\)), any reduction axis and `keepdim`, with the degenerate cases — NaN, \(\pm\infty\), all-zero segments, empty segments — settled under an explicit precedence.

## Rules

- The full signature and parameter semantics live only in the starter docstring; that is the single source of truth for judging.
- No `torch.norm` / `torch.linalg.norm` / `x.norm()` / `torch.linalg.vector_norm`, and no computing \(\sum|x_i|^p\) then taking the \(p\)-th root — that is precisely the broken version.
- The core identity, where seg is one segment to reduce and \(m=\max_i|x_i|\):

```
||x||_p = m * ( sum_i (|x_i| / m)^p ) ^ (1/p)
```

- After scaling every term lies in \([0,1]\), so the power neither overflows nor underflows; for \(p=+\infty\) just return \(m\) — a power of a max is meaningless, and that \(m\) is exactly the scale factor. \(0<p<1\) gives a quasi-norm and the identity still holds.
- Degenerate cases are decided top-down, first match wins: (1) NaN in the segment → NaN (NaN outranks inf); (2) \(\pm\infty\) in the segment → \(+\infty\); (3) \(m=0\) (all zeros) → \(0.0\), otherwise \(|x_i|/m\) becomes \(0/0\); (4) segment of length 0 (empty tensor or reduction axis of length 0) → \(0.0\); (5) everything else goes through the scaled formula.
- Validation: \(x\) must be a float32 or float64 tensor — non-tensor or non-float raises ValueError; \(p\) must be an int or float strictly greater than 0 (bool does not count; NaN, \(\le0\), \(-\infty\) and non-numeric types raise ValueError; `math.inf` is allowed); `dim=None` means reduce everything, returns a 0-D tensor and makes `keepdim` irrelevant, otherwise `dim` must be an int (negative axes allowed, and one that falls outside \([0,\text{ndim})\) after normalization raises ValueError); `keepdim` accepts bool only.
- Accumulate internally in float64 and return the input dtype; a float32 input whose result leaves float32 range may naturally become inf, but never NaN.

## Example

```
x = [[3e154, 4e154, 0.0],
     [3e-200, 4e-200, 0.0]]

p=2    -> [5e154, 5e-200]
p=1    -> [7e154, 7e-200]
p=inf  -> [4e154, 4e-200]
```

For the first row the scale is \(m=4\times10^{154}\) and \((0.75^2+1^2+0^2)^{1/2}=1.25\), which multiplies back to \(5\times10^{154}\); the second row runs the same path with \(m=4\times10^{-200}\) and lands on \(5\times10^{-200}\). Without scaling the first row's \(9\times10^{308}+16\times10^{308}\) blows up to \(+\infty\) and the second row's \(9\times10^{-400}\) vanishes to \(0\) — measured, both `torch.linalg.norm(x, dim=1)` and `torch.sqrt((x*x).sum(1))` give \([\infty,\,0]\), and `numpy.linalg.norm` behaves the same. Small hand-checkable cases: \([3,4,0]\to 5\), and \([-5,12]\to 13\) (\(p=2\)) / \(17\) (\(p=1\)) / \(12\) (\(p=\infty\)).

## Constraints

- any shape, any single reduction axis (negatives included) combined with `keepdim`; an axis of length 0 yields a correctly shaped all-zero result
- the oracle has nothing to do with scaling: `Decimal` at 60 digits accumulates \(\sum|x_i|^p\) directly and takes the \(p\)-th root, fed segment by segment
- judging uses relative tolerance: 1e-12 on the example, 1e-9 on random and special values, 1e-6 on the extreme float32 magnitudes
- running time \(O(\text{numel})\): one extra segment maximum and one extra elementwise division compared to the naive version
