# Notes: Tuned Lens Affine-Probe Ledger

## Provenance

AlignmentResearch/tuned-lens (paper 2303.08112): each probe is "LayerNorm + affine" mapping layer ℓ's residual stream into vocab space; this problem takes the **affine part** (`logits_ℓ = W_ℓ h_ℓ + b_ℓ`), with the LayerNorm recorded in the notes as a simplification (judging would need LN params as inputs too — interface kept narrow). KL/CE all in float64.

## Key points and traps

1. **KL direction**: `KL(p_final ‖ p_layer)` — p_final carries the weights (`p·(ln p − ln q)`). The reversed form is caught by both the `anchor_hand` hand value and random comparison.
2. **Temperature applies to both sides**: softmax over `logits/T`; a one-sided mutation is caught by `temperature_effect` (KL and CE must both change between T=1 and T=4). T ≤ 0 raises ValueError.
3. **CE's target** is `argmax(final_logits)` (one per batch), CE ledger shape [L,B] — gathered by target; the first draft indexed `logp[:, arange(B), target]`, treating L as the batch dim (shape error), fixed with `gather`.
4. **When KL=0**: only for the layer whose probed logits equal the final logits exactly (the last layer when the probe is the LM head); earlier layers have KL>0 — precisely the inter-layer gap tuned lens measures. The first-draft case wrongly assumed all layers get KL=0.
5. **`torch.matrix_rank` moved to `torch.linalg.matrix_rank` in torch 2.14** (param `symmetric`→`hermitian`; the old entry point errors outright) — the `probe_rank` ledger uses the new API; rank-deficient (duplicated rows) vs full-rank probes pinned by `probe_rank`.
6. **_py's float32 truncation**: the comparison helper casts float64 to float32, so 1e-9-level assertions get clipped by rounding noise — anchor tolerances need ≥1e-6 (the `ce_monotone` case tripped on this).

## Mutation testing (7/7 caught)

probe without bias, KL direction reversed, temperature dropped, CE sign dropped, argmax_match flipped, rank tol blown up, match reading only the last layer.

## Complexity

O(L·B·V·D) projection + O(L·B·V) distribution ledger, scale ≤ 4×2×9×6, milliseconds.
