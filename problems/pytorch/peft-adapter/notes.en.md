# Adapter Bottleneck Feed-Forward with Residual Explained

## Source
Houlsby et al., *Parameter-Efficient Transfer Learning for NLP* (arXiv:1902.00751). An Adapter is inserted after each Transformer sub-layer (after attention, after FFN): downsample \(d\to r\) → nonlinearity → upsample \(r\to d\) → residual. The original uses ReLU for the nonlinearity and optionally adds Dropout after the bottleneck; this problem removes Dropout to keep grading deterministic.

## Core points and pitfalls
1. **Down-projection first, up-projection second**: \(W_\text{down}:(r,d)\) first compresses \(d\) to \(r\), then ReLU, then \(W_\text{up}:(d,r)\) returns to \(d\). Reversing the shape order causes a dimension mismatch.
2. **ReLU only at the bottleneck**: the nonlinearity sits between the two projections, not on the final output or on the residual.
3. **Residual adds the original input**: \(y = u + x\), requiring \(u\) and \(x\) to share the \(d\) dimension — which is exactly why \(W_\text{up}\)'s output dim must equal \(d\).
4. **Identity starting point**: if \(W_\text{up}=0\), then \(y=x\) (the bypass leaves the backbone unchanged), which helps avoid accuracy loss right after loading.

## Grading details
The reference solution uses batched "row vector × transpose" (\(xW_\text{down}^\top\)); the grader's `_oracle` works sample by sample with column vectors \(W_\text{down}x_i\to\text{ReLU}\to W_\text{up}(\cdot)+x_i\) (a different structure, numerically equivalent), `allclose(atol=1e-5,rtol=1e-5)`. Additional cases: `W_up=0` pure residual, ReLU non-negativity (asserting that negative components of the down-projection are zeroed), and 3D batches.

## Complexity
Each Adapter has \(2rd\ll d^2\) parameters; the forward pass is two small matrix multiplications plus one element-wise ReLU and one addition.
