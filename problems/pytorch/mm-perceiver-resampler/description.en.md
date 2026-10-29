# Multimodal: Masked Perceiver Latent Resampling

## Task

Use latent queries to read visual tokens and perform **one Perceiver attention residual update**. Keys and values include the latents themselves, not just visual tokens. Attention therefore remains well-defined even when every visual token is hidden.

This is an intentionally extracted OpenFlamingo Perceiver sublayer, **not a complete PerceiverResampler, Q-Former, or OpenFlamingo model**. `visual_mask` is an exercise-specific extension absent from the original OpenFlamingo Perceiver interface. Feed-forward layers, final LayerNorm, and time/frame embeddings are excluded.

The bilingual starter contract is the single reference for parameters, shapes, numeric limits, and exceptions. The rules below define the computation.

## Rules

1. Apply non-affine LayerNorm independently to the feature dimension of each token:
   $$
   \operatorname{LN}(z)=\frac{z-\mu}{\sqrt{\frac{1}{D}\sum_i(z_i-\mu)^2+\varepsilon}},
   \qquad \mu=\frac{1}{D}\sum_i z_i.
   $$
   Normalize visual tokens and latents separately. Never mix batch, token, or head dimensions when computing statistics.
2. Queries are `LN(latents) @ wq.T`. Concatenate `LN(visual)` followed by `LN(latents)` **along the token axis**, then project separately with `wk.T` and `wv.T` to obtain keys and values. All projections are bias-free.
3. Split the feature dimension into `H` consecutive heads of width `d_h = D/H`. Compute scores as `Q_h @ K_h.T / sqrt(d_h)`.
4. Key visibility is `[visual_mask, K True entries]`, broadcast over heads and queries. Apply a numerically stable softmax over keys. Hidden positions must have exactly zero probability. Latent keys are always visible, so no query is fully masked.
5. Compute each head's weighted value sum, concatenate features in the original head order, project with `wo.T`, and add the **original, unnormalized latents**.

Return the residual result and per-head attention probabilities. Their last axis lists visual tokens first and latents second. Return probabilities, not scores or head-averaged weights.

## Example 1: Visual masking and latent keys

All floating tensors use float64. All four weights are the 2-by-2 identity matrix, `n_heads = 1`, and `eps` has its default value.

```text
visual = [[[1, -1], [-1, 1]]]
latents = [[[0, 0], [1, -1]]]
visual_mask = [[true, false]]

output = [[[0.666663, -0.666663], [1.891611, -1.891611]]]
attention = [[[[0.333333, 0, 0.333333, 0.333333],
               [0.445808, 0, 0.108385, 0.445808]]]]
```

The second visual token is hidden, but both latent keys participate in normalization. The first query is zero, so its three visible keys are equally likely.

## Example 2: No visual tokens

All floating tensors use float64, `n_heads = 1`, and `eps` has its default value.

```text
visual = empty tensor with shape (1, 0, 1)
visual_mask = empty bool tensor with shape (1, 0)
latents = [[[3]]]
wq = wk = wv = wo = [[1]]

output = [[[3]]]
attention = [[[[1]]]]
```

Single-feature LayerNorm produces zero, yet the sole latent key has probability 1. The residual preserves the original value.

All displayed outputs were dumped from actual reference execution and checked against an independent oracle; values are rounded to six decimal places.
