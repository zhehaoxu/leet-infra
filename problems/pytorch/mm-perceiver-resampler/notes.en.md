# Notes: Perceiver Latent Attention Residual

## Approach

Validate the head count first, then normalize over the full feature dimension before splitting heads. Give `torch.layer_norm` only `D` as its normalized shape and no affine parameters.

Concatenate normalized visual tokens and latents, then apply separate key/value projections. Queries come only from latents. Split features with `reshape` and swap token/head axes without assuming contiguous inputs. Append an all-True latent mask to the visual mask and broadcast it across queries and heads. Replace hidden scores with negative infinity and apply stable softmax. At least one latent key exists, so there is no fully masked row to special-case.

Swap the weighted results back to token/head order before merging features. Directly reshaping head-major data into token-major data scrambles the result. Apply `wo.T` and add the original latents, not their normalized version.

The reference uses explicit matrix multiplications, LayerNorm, and softmax; it never calls SDPA or a multi-head attention wrapper.

## Pitfalls and boundaries

- Omitting latent K/V changes the probability denominator and breaks empty visual input.
- `True` means visible; reversing mask semantics hides the wrong visual tokens.
- Never normalize across tokens/batches or use unbiased sample variance. With `D = 1`, normalization produces zero, not NaN.
- Masking every visual token does not fully mask attention: latent keys/values remain valid.
- Perturbing finite masked visual values must not affect either returned tensor.
- Zero `wv` or `wo` makes the result exactly equal to the input latents, but attention probabilities must still be computed.
- Weights follow `x @ W.T`. Nonsymmetric test weights expose missing transposes.
- Do not mutate inputs, force float32, or require contiguous storage.

## Independent validation

The oracle works batch by batch, selecting visible visual keys and all latent keys first. It normalizes using explicit means/population variances and projects through independent `F.linear` calls. Native PyTorch SDPA runs separately for each head on the selected keys/values, rather than sharing the reference's global masking path.

Attention probabilities are checked separately using per-query dot products, maximum subtraction, and exponential normalization, then scattered back to the full key axis. The 15 `check_*` functions cover examples, latent inclusion, mask direction and padding perturbations, all-hidden visual input, empty visual input, single features/latents, custom epsilon, multiple head counts, nonsymmetric weights, zero projections, both dtypes, noncontiguous inputs, invariants, and invalid head counts. A shape sweep adds multiple random seeds. `_example` and `_empty_example` construct the public examples; documented values were dumped from actual execution.

## Complexity

Let `L = N + K`. Including dense projections, time is `O(B L D² + B K L D)`. Main intermediate storage is `O(B L D + B H K L)`, excluding input weights. Returning the full attention matrix alone requires `O(B H K L)` space.

## Source and intentional simplifications

Based on the already retrieved [OpenFlamingo helpers.py](https://raw.githubusercontent.com/mlfoundations/open_flamingo/main/open_flamingo/src/helpers.py), lines 25–132, covering `PerceiverAttention` / `PerceiverResampler`. The exercise preserves separate visual/latent normalization, inclusion of latents in K/V, and the attention residual update.

The visual mask and observable attention return are exercise extensions. Latents and four separate weights are caller-provided, and LayerNorm is fixed to non-affine. Only one attention update runs: there are no internal learned/repeated latent parameters, feed-forward sublayers, full multilayer resampler, final LayerNorm, or time/frame embeddings. This is not Q-Former or a complete OpenFlamingo model.
