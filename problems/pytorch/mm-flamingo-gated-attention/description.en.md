# Multimodal: Flamingo Media Masks and Gated Fusion

## Task

In a text sequence containing media markers, let each token read **only the most recently encountered media item**. Add the resulting media attention update to the text through a signed scalar gate.

This intentionally extracts the OpenFlamingo **gated attention sublayer**, not a complete `GatedCrossAttentionBlock`, Q-Former, or OpenFlamingo model. The feed-forward sublayer and its second gate are excluded. There is no cache; semantics are fixed to `only_attend_immediate_media=True`.

The bilingual starter contract is the single reference for parameters, shapes, numeric limits, and exceptions. The rules below define the computation.

## Media visibility

A True entry in `media_locations` marks a media occurrence in the text; it is **not a padding mask**. The first marker corresponds to the first media item, the second marker to the second item, and so on.

Define `text_time = cumsum(media_locations)`. Media items have logical IDs 1 through M. With zero-based array indices:

$$
\mathrm{visible}[b,t,m,k] = \bigl(\mathrm{text\_time}[b,t] = m+1\bigr).
$$

Flatten visibility and media tokens in media-major, then latent-major order. Both must use the same ordering.

- A marker **itself** can immediately read its corresponding media item.
- Between markers, only the latest item is visible; visibility does not accumulate across previous items.
- Before the first marker, or in a row with no markers, no media is visible.
- Unreferenced media cannot affect the result. Once the next item appears, the previous one must not leak into attention.

## Attention and gating

1. Apply non-affine LayerNorm only to each text token's full feature dimension, using population variance: the denominator is `sqrt(mean((x - mean(x))²) + eps)`.
2. Queries are `LN(x) @ wq.T`. **Do not normalize media.** Flatten its media and latent axes, then project separately with `wk.T` and `wv.T`. All projections are bias-free.
3. Split features into consecutive heads and scale scores by `sqrt(D/H)`. Apply stable softmax over visible media tokens only. All other probabilities must be exactly zero.
4. Rows without visible media must have **exactly zero** attention probabilities and weighted value results, without NaNs. A fully masked row must not become a uniform distribution.
5. Concatenate weighted results in the original head order, project with `wo.T`, and compute:
   $$
   \mathrm{out}=x+\tanh(\mathrm{gate})\,\mathrm{projected}.
   $$

This is a **tanh scalar gate** shared across batches, tokens, heads, and channels, not a per-channel gate or sigmoid gate. Zero gives an identity mapping. A positive gate adds the update in its computed direction; a negative gate reverses that update. Changing the gate does not change probabilities or visibility.

Return the residual result, ungated per-head attention probabilities, and flattened boolean visibility. Before the first media item, the output must exactly equal the original text features.

## Example 1: Immediate visibility, latest media only

All floating tensors use float64. All four weights are the 2-by-2 identity matrix, `n_heads = 1`, and `eps` has its default value. Set `gate = log(3)/2 ≈ 0.5493061443340549`, so mathematically `tanh(gate) = 0.5`.

```text
x = [[[1, -1], [0, 0], [0, 0], [0, 0]]]
media = [[[[2, 0], [0, 2]],
          [[4, 0], [0, 4]],
          [[9, 9], [9, 9]]]]
media_locations = [[false, true, false, true]]

out = [[[1, -1], [0.5, 0.5], [0.5, 0.5], [1, 1]]]
attention = [[[[0, 0, 0, 0, 0, 0],
               [0.5, 0.5, 0, 0, 0, 0],
               [0.5, 0.5, 0, 0, 0, 0],
               [0, 0, 0.5, 0.5, 0, 0]]]]
visible = [[[false, false, false, false, false, false],
            [true, true, false, false, false, false],
            [true, true, false, false, false, false],
            [false, false, true, true, false, false]]]
```

The first token has no media. The second and third tokens read only the first item. The fourth token's marker immediately switches to the second item. The third item is never referenced.

## Example 2: A negative scalar gate

Again use float64, 2-by-2 identity weights, one head, and default `eps`, but set `gate = -log(3)/2`.

```text
x = [[[2, -2], [0, 0]]]
media = [[[[2, 4]]]]
media_locations = [[false, true]]

out = [[[2, -2], [-1, -2]]]
attention = [[[[0], [1]]]]
visible = [[[false], [true]]]
```

The negative gate reverses only the media update, not probabilities or the pre-media identity mapping.

All displayed outputs were dumped from actual reference execution and checked against an independent oracle; values are rounded to six decimal places.
