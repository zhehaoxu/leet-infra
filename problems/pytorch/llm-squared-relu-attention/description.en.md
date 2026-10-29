# Squared-ReLU Attention: relu² Instead of Softmax with Row Normalization

## Description

Replacing the attention softmax with a point-wise activation removes the whole-row reduction, which makes attention trivially tileable. Shen et al. (arXiv 2309.08586) compare the point-wise attention family \( \phi=L^{-\alpha}h \) with \(h\in\{\mathrm{relu},\mathrm{relu}^2,\dots\}\), where **squared ReLU** comes closest to softmax behaviour on vision Transformers.

Implement the full relu² attention forward pass: row-normalize the activated scores (with an `eps` floor on the denominator) and optionally apply an RMSNorm over the head dimension to q and k before the dot product (QK-Norm). The differences from softmax attention go beyond a swapped activation — the rules below spell them out.

## Rules

- The full signature and parameter meanings live only in the starter docstring; that is the sole contract.
- \( \text{logits}=q k^{\mathsf T}/\sqrt{h_d} \), then \( a=\mathrm{relu}(\text{logits})^2 \). **Apply relu before squaring**; the other order makes negatives positive and counts them as contributions.
- In `mask`, True = keep and False = zero the activated score; masked positions always get weight 0.
- \( \text{denom}=\sum_j a_j \), \( \text{weights}=a/\max(\text{denom},\texttt{eps}) \), \( \text{output}=\text{weights}\cdot v \).
- With `qk_norm=True`, first apply a **gain-free** RMSNorm over the head dimension to q and k (\(x/\sqrt{\mathrm{mean}(x^2)+\texttt{eps}}\)) and then take the dot product; `eps` doubles as the denominator floor.
- Return `(output, weights)`: `output` has shape `(..., L, v_d)`, `weights` has shape `(..., L, S)`, both float32; compute internally in float64.
- When the denominator is 0 (no positive logit in the row, or the whole row is masked) that row's weights are all 0 and its output is 0, never NaN; for \(0<\text{denom}<\texttt{eps}\) the row is scaled down uniformly and its sum is below 1.
- Leading dimensions follow broadcasting rules, e.g. q `(2, 3, L, h_d)` with k and v `(2, 1, S, ·)`.
- **Note**: relu² is homogeneous of degree two, so after row normalization the \(\sqrt{h_d}\) temperature cancels exactly — including it or not gives identical results. Likewise, multiplying q or k by a positive constant leaves the weights unchanged. This is a property of the attention family, not a bug.

## Example

```
query = [[1, 0], [1, 1]]
key   = [[1, 0], [0, 1], [-1, 0]]
value = [[1, 0], [0, 1], [5, 5]]

weights = [[1.0, 0.0, 0.0],
           [0.5, 0.5, 0.0]]
output  = [[1.0, 0.0],
           [0.5, 0.5]]
```

The first row has logits \([\frac{1}{\sqrt2},0,-\frac{1}{\sqrt2}]\); after relu² only the first entry is nonzero, so the weights are \([1,0,0]\) and the output equals the first value vector. The second row has logits \([\frac{1}{\sqrt2},\frac{1}{\sqrt2},-\frac{1}{\sqrt2}]\) and the first two entries get 0.5 each — note that the second logit is exactly 0, which softmax would score at \(1/3\) while relu² gives it exactly 0.

## Constraints

- `query`, `key`, `value` must be float32/float64 tensors with `ndim >= 2`, a non-empty head dimension, and all entries finite; NaN or ±inf raises ValueError.
- q and k must share the head dimension; k and v must share all leading dimensions exactly; non-broadcastable leading dimensions raise ValueError.
- `mask` must be a bool tensor broadcastable to the score matrix; `qk_norm` must be a bool; `eps` must be a finite positive real (bool, 0, negatives and NaN raise ValueError).
- Time is \(O(\text{batch}\cdot\text{heads}\cdot L\cdot S\cdot h_d)\) and the weight matrix takes \(O(\text{batch}\cdot\text{heads}\cdot L\cdot S)\).
