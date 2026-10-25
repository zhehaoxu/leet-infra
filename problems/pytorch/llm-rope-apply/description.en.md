# LLM Component: RoPE Rotary Position Embedding

## Problem Description

**RoPE (Rotary Position Embedding)** is the position encoding shared by LLaMA / Qwen / Mistral / DeepSeek. Instead of *adding* a position vector to the input, it **rotates** each pair of adjacent dimensions of query/key by an **angle** determined by the position $m$, so that the attention dot product depends only on the **relative position** $m-n$.

## Rules

Input: query tensor $q$ (shape $(B, H_q, S, d)$), key tensor $k$ (shape $(B, H_k, S, d)$), position indices `position_ids` (shape $(B, S)$, integer), head dimension $d$ and base frequency $base$; return the **rotated** $(q_{rot}, k_{rot})$, each keeping its shape. Steps:

1. Half-dimension inverse frequencies ($i = 0 \dots d/2-1$):
   $$ \text{inv\_freq}_i = base^{-\,2i/d} $$
2. Angles and cos/sin: for each position $m$, $\theta = m \cdot \text{inv\_freq}$, then duplicate along the last dimension to form $d$ dimensions
   $$ \text{emb} = [\theta \,\Vert\, \theta],\quad \cos = \cos(\text{emb}),\ \sin = \sin(\text{emb}) \quad (\text{shape } (B,S,d)) $$
3. Apply the rotation (`rotate_half` treats $x=[x_1,x_2]$ as two halves, with $[x_1,x_2]\mapsto[-x_2,x_1]$):
   $$ q_{rot} = q \odot \cos + \text{rotate\_half}(q) \odot \sin $$
   $k$ likewise. cos/sin must broadcast over the head dimension (that is, expand $(B,S,d)$ along the head axis).

> The judge environment has `transformers` installed: submissions are cross-checked against
> `Qwen2RotaryEmbedding` + `apply_rotary_pos_emb`
> (same `position_ids`, same $d$, same $base$). cos/sin are computed in **float32**.

## Example

```
d = 4, base = 100, position_ids = [[1]], q = [1, 2, 3, 4]
inv_freq = [100^0, 100^-0.5] = [1.0, 0.1]
θ = [1.0, 0.1] → emb = [1.0, 0.1, 1.0, 0.1]
q_rot ≈ [-1.984111, 1.590675, 2.462378, 4.179684]
```

## Constraints

- $d$ is even; $q$ has shape $(B,H_q,S,d)$ and $k$ has shape $(B,H_k,S,d)$ (under GQA, $H_q \ne H_k$ requires correct broadcasting for each)
- `position_ids` has shape $(B,S)$ with integer dtype
- Return the 2-tuple `(q_rot, k_rot)`, with shapes matching the inputs and dtype float32
- `torch.allclose` with relative tolerance $10^{-5}$ against the `transformers` reference implementation; do not modify the inputs
