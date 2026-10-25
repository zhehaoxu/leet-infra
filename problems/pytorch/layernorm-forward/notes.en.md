# LayerNorm (Forward) in Depth

## Definition

Normalize **each sample** along the feature dimensions, independent of the batch:

$$ \hat{x}=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}},\qquad y=\gamma\hat{x}+\beta $$

$\mu,\sigma$ are computed **per sample** over the normalized dimension (usually the hidden dimension `d_model` in a Transformer).

## The essential difference from BatchNorm

- BN computes statistics along the batch, shares them per channel, needs running stats, and depends on the batch size; LN computes statistics **per sample** along the features, is independent of the batch size, uses the same path in training and inference, and has no running stats.
- That is why RNNs/Transformers (variable-length sequences, single-sample inference) use LN rather than BN.

## Dimension pitfalls (the most error-prone part here)

- For $(B,T,C)$, normalizing the last dimension `dim=-1` gives $\mu,\sigma$ of shape $(B,T,1)$, while $\gamma,\beta$ of shape $(C,)$ broadcast over the last dimension.
- The reduction dimensions must line up with `normalized_shape`; in multimodal settings several consecutive dims are sometimes normalized (e.g. the last two)—follow the dimensions given by the problem.
- $\epsilon$ goes **inside** the square root (as in BN); the default is `1e-5`.

## Relation to RMSNorm

Subtracting the mean is LN's "centering" step; RMSNorm drops the mean subtraction and only divides by the root mean square, which is cheaper. Once you understand LN, RMSNorm is simply its simplified version.

## Reference for cross-checking

`F.layer_norm(x, normalized_shape, weight, bias, eps)`, where `normalized_shape` is the shape tuple of the last dimension (or last few dimensions).
