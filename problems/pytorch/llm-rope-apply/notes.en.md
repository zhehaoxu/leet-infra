# RoPE in Detail

## Core Idea: Encoding Relative Position Through Rotation

Ordinary absolute position encoding *adds* a vector; RoPE instead applies a **2D rotation** to each pair of adjacent components of query/key, with the rotation angle growing linearly with position $m$. That way, when two vectors are dotted, the absolute angles cancel and only the **relative offset** $m-n$ contributes:

$$ \langle R_m\,q,\; R_n\,k\rangle = \langle q,\; R_{n-m} k\rangle $$

This is exactly what large models favor: attention naturally sees how far apart tokens are, and it has extrapolation potential for longer sequences.

## Implementation Notes

- **Half-dimension frequencies**: only the first $d/2$ frequencies $\text{inv\_freq}_i=base^{-2i/d}$ are independent; the angles are duplicated into $d$ dimensions with `cat((θ, θ))` because `rotate_half` views the $d$ dimensions as two halves $(x_1,x_2)$ paired for rotation.
- **rotate_half rather than interleaving**: HF/LLaMA pair the dimensions front-half with back-half (`[-x2, x1]`), not adjacent interleaved pairs. Get the pairing wrong and the relative-position property of the dot product breaks - this is also where different implementations most often fail to agree.
- **Broadcasting**: cos/sin have shape $(B,S,d)$ and must be `unsqueeze(1)`-ed along the head axis into $(B,1,S,d)$ so they can feed both the $H_q$ query heads and the $H_k$ key heads (whose counts differ under GQA).
- **Compute trig in float32**: when angles are large, low-precision cos/sin errors are noticeable, so HF computes in fp32 and then casts back down.

## Base Frequency and Long Context

The larger $base$ is, the longer the wavelength of the lowest-frequency component, and the farther the relative distance it can represent. Qwen/LLaMA commonly expand base from 10 thousand to 100 thousand or even a million to support long context (combined with scalings such as YaRN; see the rope-scaling problems).

## Related

- Upstream: `llm-rmsnorm`.
- Downstream: in `llm-gqa-attention`, the rotation from here is called right after the q/k projections; `rope-embedding` (operators collection) is an earlier simplified version.
- Cross-check: `transformers`' `Qwen2RotaryEmbedding` + `apply_rotary_pos_emb` (same position_ids/d/base), `allclose` $10^{-5}$.
