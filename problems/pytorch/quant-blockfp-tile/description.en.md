# Quantization: BlockFP 1×128 Tile and UE8M0 Shared Exponent

## Description

**BlockFP** (the tile quantization format used in DeepSeek-V3 technical report §3.1) slices a matrix into **1×128 tiles** along the last dimension, with each tile sharing one **UE8M0** exponent. UE8M0 is isomorphic to E8M0: 8-bit unsigned exponent, bias 127; code 0 means scale 0, otherwise

\[\text{scale}=2^{\text{code}-127}.\]

Given \(W\in\mathbb R^{M\times N}\) (\(N\bmod 128=0\)), split row-major into tiles of 128 columns each:

1. Compute the tile maximum absolute value \(m_t\).
2. If \(m_t=0\), then \(\text{code}_t=0\) and \(\text{scale}_t=0\).
3. Otherwise
   \[e_t=\left\lceil\log_2 m_t\right\rceil,\quad \text{code}_t=\operatorname{clip}(e_t+127, 1, 255),\quad \text{scale}_t=2^{\text{code}_t-127}.\]
4. The normalized tile values are \(\hat W=W/\text{scale}_t\).

## Task

Implement `solve(W)`:

- Input: `W` (float32 \((M, N)\), \(N\bmod 128=0\)).
- Return `(scale_codes, normalized)`:
  - `scale_codes` is uint8 with shape \((M, N/128)\);
  - `normalized` is float32 with the same shape as `W`.

## Hints

- The main difference from MX Microscaling is the tile size and format origin (BlockFP / DeepSeek-V3); the encoding rule is the same.
- Avoid NaNs from division by zero for all-zero tiles.
