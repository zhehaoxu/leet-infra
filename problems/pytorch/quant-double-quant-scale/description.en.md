# Quantization: Second-Level Quantization of Scales (Double Quantization)

## Problem

Block quantization stores one FP32 scale per block (QLoRA uses block=64, i.e. 0.5 bit/param of overhead; on a 65B model that alone is GB-scale). **Double quantization** (Dettmers et al., NeurIPS 2023, arXiv:2305.14314 §3) **quantizes the scales themselves again**: a group of `block_size` first-level scales shares one FP32 second-level scale, and first-level scales are stored as 8-bit fixed point.

This problem implements the numerical core of that "scale of scales" (byte accounting lives in `peft-qlora-nf4-forward`). Let the first-level scale array be \(s\in\mathbb R^{B}\), split into **consecutive** groups of `block_size` (default 256; a short trailing group forms its own group):

\[
s^{(2)}_g=\frac{\max_{b\in g}|s_b|}{2^{k-1}-1}\quad(\text{if }\max=0,\ s^{(2)}_g=1),\qquad
c_b=\operatorname{clamp}\!\left(\operatorname{round}\!\left(\frac{s_b}{s^{(2)}_g}\right),-(2^{k-1}-1),\,2^{k-1}-1\right),\qquad
\hat s_b=c_b\,s^{(2)}_g
\]

That is **symmetric fixed-point** quantization: `k` signed bits, `qmax = 2^(k-1)-1`, and the largest-magnitude scale in each group lands exactly on \(\pm qmax\).

## Task

Implement `solve(scales, bits, block_size)`:

- Input: `scales` float32 `(B,)` (first-level scales, all positive in practice but not validated), `bits` an integer in 2..8, `block_size` a positive integer.
- Group consecutively by `block_size` (the last group may be short); compute `s2`, quantize and dequantize per group.
- Output `(codes, group_scale, deq)`: `codes` int64 `(B,)`; `group_scale` float32 `(n_groups,)`; `deq` float32 `(B,)`.

**Conventions**: float64 internally, float32 output; tolerance \(10^{-5}\). Out-of-range `bits` or `block_size <= 0` raises `ValueError`.

## Example

```
scales = [0.25, 0.25, 0.25, 0.25], bits = 2, block_size = 2
qmax = 1; group 0: s2 = 0.25/1 = 0.25 → codes [1,1], deq [0.25,0.25]
group 1 likewise
Output: codes=[1,1,1,1], group_scale=[0.25,0.25], deq=scales (exact)
```

## Constraints

- Group boundaries are "consecutive block_size elements", not value-based buckets; a short trailing group **stands alone** and never borrows another group's scale.
- The largest-magnitude element of each group must be reconstructed exactly (error ≤ 1e-6).
- An all-zero group gets `s2 = 1.0`, all codes 0, reconstruction 0 (avoid division by zero).
