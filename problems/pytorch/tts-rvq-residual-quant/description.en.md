# RVQ Residual Vector Quantization Ledger

## Problem

Neural audio codecs (EnCodec / SoundStream / Mimi / DAC — the same core behind llama.cpp's Qwen3-TTS and Pocket-TTS code-to-wave) encode continuous vectors as per-level codebook indices via **RVQ** (Residual Vector Quantization): level k quantizes the previous level's residual by nearest neighbor, shrinking the residual level by level. This problem transcribes that quantization ledger: given `[N, D]` vectors and a `[K, C, D]` per-level codebook stack, output the per-level code indices, the reconstructed vectors, and the per-level residual energies.

## Solution

The ledger is pinned by the spec (independent per sample):

- `residual = latent`; for each level `k = 0..K-1`:
  - `idx = argmin_c ||residual - codebooks[k][c]||²` (**ties pick the smallest index**);
  - `codes[k][n] = idx`; `residual -= codebooks[k][idx]` (residual shrinks level by level);
  - `residual_sq[k][n]` = the L2² of the residual **after** this level's subtraction;
- `reconstructed[n]` = the sum of the selected codebook entries across levels (accumulated in increasing-level order; float order fixed).

Boundary with the quantization collection: that covers scalar weight/activation quantization and codebook learning; this is the **residual-cascade encoding structure**. Returns `{"codes", "reconstructed", "residual_sq"}`.

## Constraints

- 1 ≤ N ≤ 10^4, 1 ≤ D ≤ 512, 1 ≤ K ≤ 32, 1 ≤ C ≤ 4096.
- Empty latents/codebooks, dim mismatches (between latents, or codebook entries vs D), an empty codebook level, non-numeric or NaN/Inf, bool → `ValueError`.
