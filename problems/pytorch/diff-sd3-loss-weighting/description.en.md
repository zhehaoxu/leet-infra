# Training Losses: SD3 Sigma Weighting and Confidence-Aware Cross-Entropy

## Problem

A single Flow Matching training step has two "loss side" books to balance: one assigns a per-sample weight to the error according to the noise level σ (the σ-sqrt / cosmap roster from the SD3 paper), and the other adds a regularization term on a token-classification head that only shrinks the entropy of tokens the model already predicts correctly. Both live in diffusers' `training_utils.py`.

## Rules

Part one, σ weighting: the input is a floating-point tensor `sigmas` (finite positive entries) plus a string `weighting_scheme`.

- `"sigma_sqrt"`: the weight is \(\sigma^{-2}\) and is **always cast to float32** (upstream has a `.float()` right there), so a float64 σ comes back as float32.
- `"cosmap"`: the weight is \(2/(\pi(1-2\sigma+2\sigma^2))\), with the **dtype following `sigmas`**.
- Any other string (including `"none"`, the empty string, a different case, or the typo `"cos_map"`): return an all-ones `ones_like` in the same dtype, **without raising**.

Part two, the confidence-aware loss: `logits` has shape \((...,V)\) and `labels` is an integer tensor of shape \((...)\).

- `valid` \(=(\text{labels}\ne\text{ignore\_index})\); ignored positions enter neither the numerator nor the denominator.
- `weights` is `per_token_weights` or all ones, always cast to `logits.dtype`.
- `loss_sft` \(=\sum \text{nll}\cdot\text{weights}\cdot\text{valid}\big/\operatorname{clamp\_min}(\sum\text{weights}\cdot\text{valid},1)\), where nll is the per-token cross-entropy.
- When `lambda_conf == 0` return `(loss_sft, loss_sft, 0)` directly, without computing any entropy.
- Otherwise set `pred = argmax(logits)` and `correct = valid ∧ (pred == labels)`, then
  \(p=\operatorname{softmax}(\text{logits.float}()/\text{temperature})\) (**the temperature only touches the entropy, never the CE**),
  \(H=-\sum p\log\operatorname{clamp\_min}(p,\text{tiny})\),
  `loss_conf` \(=\sum H\cdot\text{weights}\cdot\text{correct}\big/\operatorname{clamp\_min}(\sum\text{weights}\cdot\text{correct},1)\),
  and `loss` \(=\) `loss_sft` \(+\) `lambda_conf` \(\cdot\) `loss_conf`.
- Both denominators pass through `clamp_min(1)`: when the weights are all below one the result is not a weighted average but a sum truncated at 1.

Return `(weighting, loss, loss_sft, loss_conf)`: `weighting` has the shape of `sigmas`; the last three are 0-dim tensors in `logits`' dtype.

## Example

`sigmas = [0.1, 0.5, 1.0]` (float32):

```
sigma_sqrt -> [99.999992, 4.000000, 1.000000]   # the float32 inverse square of 0.1 is not exactly 100
cosmap     -> [0.776366, 1.273239, 0.636620]    # 1.273239 = 4/π, maximal at σ=0.5
none       -> [1.000000, 1.000000, 1.000000]
```

The three rows of `logits` are `[[1,3,0], [0,1,2], [2,0,1]]`, `labels = [1, -100, 1]`, `lambda_conf=0.5`, `temperature=2`:

```
loss = 1.741706, loss_sft = 1.288726, loss_conf = 0.905959
lambda_conf=0 -> loss = loss_sft = 1.288726, loss_conf = 0.000000
```

The second position is dropped by `ignore_index`; the third position has label 1 while its argmax is 0, so it enters the CE only and not the confidence term — the latter looks at the first position, which is the only hit.

## Constraints

- `sigmas` must be a floating-point tensor, non-empty, with all entries finite and positive; σ=0 would blow `sigma_sqrt` up to inf and counts as invalid input.
- `logits` has at least 2 dims, is floating point, and has a non-empty last dim; `labels` must be an integer tensor whose shape equals `logits.shape[:-1]`.
- Every `labels` entry that differs from `ignore_index` must lie in \([0,V)\), where \(V\) is the last dim of `logits`.
- `temperature > 0`; `lambda_conf` must be a finite number; `per_token_weights` must be a floating-point tensor shaped like `labels`.
- Any violation of the above raises `ValueError`.
