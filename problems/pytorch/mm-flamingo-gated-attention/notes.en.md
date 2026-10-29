# Notes: Immediate-Media Masking and Scalar Gating

## Approach

Validate the head count first. Convert media markers to integers, take a cumulative sum along text, and compare against media IDs starting at 1. Do not use `>=`, which would reveal all previous media. Do not subtract one from the cumulative sum before comparison, which would hide the new media from its own marker. Repeat each item's visibility over all its latents, then flatten in media-major order.

Normalize text features only, then project queries. Keep media features unnormalized and project keys/values directly after flattening. Use `reshape` and axis swaps to split heads while accepting noncontiguous tensors.

## Safe fully masked rows

For ordinary rows, set hidden scores to negative infinity. For rows with no media, replace scores with finite zeros **before** softmax, then set every hidden probability to zero afterward. Applying softmax to all negative infinities and then multiplying by a zero mask is unsafe: NaN times zero is still NaN.

With this exercise's finite inputs, exactly zero probabilities produce exactly zero weighted values. Merge heads, apply the bias-free output projection, and add the update multiplied by `tanh(gate)`. Tokens without media retain their original residual. Returned probabilities are not gated; a zero gate still requires correct probabilities and visibility.

The reference uses explicit matrix multiplications and stable softmax, not SDPA or a multi-head attention wrapper.

## Gate semantics

`gate` is a shared real scalar; `tanh` bounds its coefficient to [-1,1]. Zero initialization makes this sublayer's forward mapping exactly the identity, which allows a new branch to be introduced into a pretrained text network. It does not imply zero attention probabilities. Opposite gate values produce opposite residual updates. This is neither a separately learned coefficient per channel nor a sigmoid gate in [0,1].

## Independent validation

The oracle counts markers with Python control flow, independently for each batch and token, then selects the **raw slice of the sole visible media item**. With no marker, it leaves probabilities zero and preserves the original text. Otherwise, it invokes native PyTorch SDPA on that selected slice separately for each head. It does not construct the reference's global masked-softmax. Text normalization uses explicit means/population variances and projections use `F.linear`. Probabilities are computed separately through per-head dot products, maximum subtraction, and exponential normalization, then placed into the full media axis.

The 18 `check_*` functions cover two executed examples, zero-gate identity, positive/negative gates and ±20, immediate marker visibility, no markers, batch-dependent pre-media prefixes, isolation from previous media, perturbations of unreferenced media, unnormalized media, per-token LayerNorm, custom epsilon, nonsymmetric weights, multiple heads, minimal dimensions, zero projections, float32/64, noncontiguous tensors, input preservation, and invalid heads. Stability tests construct logits near ±1000 in both dtypes while also including fully masked rows, checking finite outputs and exact zero probabilities. A shape sweep adds multiple random seeds.

The public examples come from `_example` and `_negative_example`; documented values were dumped from actual execution after independent oracle validation.

## Complexity

Let `L = M K`. The vectorized implementation takes `O(B (T+L) D² + B T L D)` time and `O(B (T+L) D + B H T L)` main intermediate space, excluding input weights. Although each token reads at most one media item, the returned probabilities cover the full media axis, requiring `O(B H T L)` output space.

## Source and scope

Based on the already retrieved [OpenFlamingo helpers.py](https://raw.githubusercontent.com/mlfoundations/open_flamingo/main/open_flamingo/src/helpers.py), lines 136–279, covering `MaskedCrossAttention` / `GatedCrossAttentionBlock`. The exercise preserves cumulative media-marker counting, immediate-media visibility, zero attention before the first media item, and the tanh-gated residual.

The exercise fixes no-cache, immediate-media-only behavior. LayerNorm is non-affine, weights are caller-provided, and probabilities/visibility are additionally returned for inspection. Only the gated attention sublayer is extracted: the original block's feed-forward branch and its second gate are excluded, as are the complete visual encoder, language model, and multilayer fusion. This tanh scalar gate is not per-channel gating, and the exercise is not a complete Q-Former/OpenFlamingo model.
