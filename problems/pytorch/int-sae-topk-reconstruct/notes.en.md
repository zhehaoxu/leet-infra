# Notes: SAE Top-k Encoding Ledger

## Provenance

`jbloomAus/SAELens` TopK SAE architecture (`sae/sae.py` `encode`/`decode`: `pre_acts = W_enc(x − b_dec) + b_enc`, then top-k **without ReLU**; `decode = acts @ W_dec + b_dec`). Hidden states and weights are all explicit inputs, honoring the "no real-model activations" red line. The JumpReLU-threshold variant (fixed θ replacing the top-k selection, the rest of the ledger isomorphic) is noted here but not issued as a separate problem.

## Key points and traps

1. **No ReLU is the boundary between TopK SAE and ReSAE/plain SAE**: negative pre-acts are selected as-is and participate in reconstruction — both "clamp after top-k" and "clamp before top-k" mutations are caught by `negative_pre_acts_kept` plus random comparison.
2. **b_dec's double role**: subtracted at encode (`x − b_dec`) and added back at decode — dropping either side is caught by the `k_equal_F` identity assertions (at k=F, acts = full pre_acts and recon = pre@W_dec + b_dec) and random comparison.
3. **Tie-break**: `torch.topk` takes the smaller index on equal values; the oracle replicates with `sorted(key=(-pre, f))`. `dead_features` pins `[0,1]` with four equal 1.0 pre-acts.
4. **l0's boundary**: `min(k, F)` (at k=F everything is nonzero — unless some pre-act is exactly 0); `max_feat` is by **absolute** value (the largest-magnitude negative activation counts), unlike signed `argmax` — the `abs-argmax-tie` mutation needs random negative activations.
5. **mse averages over D_in** (not sum) — the 0.5 anchor in `anchor_toy` pins it.

## Mutation testing (8/8 caught)

encode without `−b_dec`, ReLU after top-k, ReLU before top-k, decode without `+b_dec`, k halved, mse as sum, max_feat without abs, recon_error sign flip.

## Complexity

O(B·F·D) encode + O(B·F·D) decode, scale ≤ 2×10×6, milliseconds.
