# notes — tts-rvq-residual-quant (order 673)

## Source

- Origin: `facebookresearch/encodec` `quantization/core_vq.py` (ResidualVectorQuantization: per-level residual `encode`; SoundStream 2107.03334 is isomorphic; Mimi/DAC/Qwen3-TTS code-to-wave shares this core). This problem takes the **inference-side per-level nearest-neighbor ledger** (no codebook learning/EMA/commitment loss).
- Ledger cruxes: nearest neighbor uses **L2²** (same ranking as L2, no sqrt); ties pick the smallest index (torch.argmin semantics); the level's residual_sq is computed **after** subtracting the selected entry (this level's absorption is deducted); reconstruction = sum of selected entries in increasing-level order.
- Boundary with the quantization collection (scalar quant/dequant, codebook training) and with #110's Qwen2-Audio token expansion (a token-count ledger).
- llama.cpp connection: Qwen3-TTS and Pocket-TTS RVQ decode is the inverse (codes → reconstructed) of this problem's encoding ledger.

## Judging

- solve args: latents [N,D], codebooks [K,C,D]. Returns codes [K][N], reconstructed [N,D], residual_sq [K][N].
- Oracle: `test_cases._py` is an independent implementation (min+index selection, reconstruction assembled separately), 150 random cross-checks (N≤5, D≤3, K≤4, C≤5), codes strict + floats at 1e-9.
- Anchors: a hand-computed 1-D two-level case (1.05 → level0 picks 1.0, residual 0.05 → level1 zeroes it; the float residual ~1.7e-33 checked at 1e-30 tolerance); level-cascade direction correction; equidistant tie picks the smaller index; multi-sample independence; a 2-D orthogonal codebook hand-computed; residual energy strictly decreasing across levels.
- Invalid inputs: empty latents/codebooks, an empty level, dim mismatches, non-numeric/NaN/Inf/bool → ValueError.

## Mutation log (9/9 caught)

tie-max-index (ties pick the larger index), no-residual-update (residual never updated), level-reverse (level order reversed), dist-no-sq (L2² → L1), resid-no-update-sq (residual_sq always 0), recon-minus (reconstruction becomes subtraction), codes-order-swap (level outputs swapped), dist-use-recon (distance drops the codebook term), first-level-skip (level 0 skipped).

## Red lines

No codebook learning/EMA/kmeans (codebooks are explicit inputs); no real audio IO or waveform synthesis; boundary with the quantization collection (scalar quant) and with #110's audio token expansion.
