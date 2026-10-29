# SoftMax1 (LLN): Attention Sink as an Extra Denominator Term

## Description

StreamingLLM (Xiao et al., ICLR 2024, arXiv 2309.17453), Section 3.3, observes that a large share of attention in LLMs lands on the first few tokens, forming an **attention sink**. Their **SoftMax1** (also called LLN, less-normalized) simply adds one to the denominator:

\[
\text{SoftMax1}(x)_i=\frac{e^{x_i}}{1+\sum_{j=1}^{N}e^{x_j}}
\]

The extra 1 is \(\exp(0)\), as if one more key with logit fixed at 0 and an all-zero value were appended. Rows therefore no longer sum to one: the row sum is below one, and the deficit is exactly the mass absorbed by that sink. Implement this normalizer with a configurable sink logit and a mask, and return both pieces of mass.

## Rules

- The full signature and parameter meanings live only in the starter docstring; that is the sole contract.
- In `mask`, True = keep and False = drop; dropped positions must have weight exactly 0 and must take part in neither the maximum nor the denominator.
- Numerically stable: per row use \(r=\max(\max_{\text{kept}} x,\ \texttt{sink\_logit})\), so every `exp` argument is \(\le 0\). The sink must join the maximum; otherwise a row far below the sink produces \(\exp(+\text{large})\).
- A fully masked row is legal input: `weights` are all 0 and `sink_weight` is 1; never return NaN.
- Return `(weights, sink_weight)`: `weights` has the shape of `scores`, `sink_weight` has shape `scores.shape[:-1]`, both float32; compute internally in float64.
- Row-wise the identity `weights.sum(-1) + sink_weight == 1` holds, hence `weights.sum(-1) < 1`, and the deficit is the mass the sink takes. Do **not** divide `weights` by its own row sum to restore 1.
- With a very negative `sink_logit` such as \(-80\), the result must degenerate to ordinary masked softmax (`sink_weight ≈ 0`).

## Example

```
scores = [[ 0.000000,  0.693147],
          [ 0.000000, -3.000000],
          [-1.000000, -2.000000]]
mask   = [[True, True], [True, False], [False, False]]
sink_logit = 0.0

weights     = [[0.250000, 0.500000],
               [0.500000, 0.000000],
               [0.000000, 0.000000]]
sink_weight = [0.250000, 0.500000, 1.000000]
```

Row 0 has \(r=\ln 2\), kept exponentials \(0.5+1=1.5\), a sink term \(e^{-\ln 2}=0.5\), hence denominator 2 and weights \([0.25,0.5]\) with sink weight 0.25. Row 1 keeps only the first entry: \(e^{0}=1\) plus the sink's 1 gives 0.5/0.5 and an exact 0 in the second slot. Row 2 is fully masked, so the sink owns all of the mass.

## Constraints

- `scores` must be a float32/float64 tensor with `ndim >= 1`, last dimension \(\ge 1\), and every entry finite; NaN or \(\pm\infty\) raises ValueError (use `mask` to exclude positions).
- `mask` must be a bool tensor broadcastable to `scores`; a non-tensor, float/integer tensor, or incompatible shape raises ValueError.
- `sink_logit` must be a finite real number; bool, string, None, NaN and \(\pm\infty\) raise ValueError (integers are fine).
- Time and space complexity are both \(O(\text{number of elements})\).
