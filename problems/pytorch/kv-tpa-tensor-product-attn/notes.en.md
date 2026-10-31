# Walkthrough: TPA Tensor-Product Attention Factor Contraction and Masking

## In one sentence

TPA splits each token's K/V into a tensor product of "per-head coefficients × head-shared basis". At decode it first computes the small `bq·bk` tile and contracts, never materializing dense per-head K. This problem reproduces its CPU semantics: sigmoid gating, four scales, and three mask forms.

## Algorithm steps

Given `aq (B,N,H,R)`, `ak (B,M,H,S)`, `av (B,M,H,S_V)`, `bq (B,N,R,D)`, `bk (B,M,D,S)`, `bv (B,M,E,S_V)`:

1. Gate: \(\tilde a=\sigma(a)\) element-wise;
2. Contract into q/k/v:

\[
q=\tilde a^q\cdot b^q\cdot s_q\ (B,N,H,D),\quad
k=\tilde a^k\cdot b^k\cdot s_k\ (B,M,H,D),\quad
v=\tilde a^v\cdot b^v\cdot s_v\ (B,M,H,E)
\]

3. Scores \(S=q\cdot k\cdot s+\text{bias}\) of shape `(B,H,N,M)`;
4. `prob = softmax(S, dim=M)`, `out = prob · v` of shape `(B,N,H,E)`.

Defaults are \(s_q=1/R\), \(s_k=1/S\), \(s_v=1/S_V\), \(s=D^{-1/2}\). `is_causal` and `attn_mask` are mutually exclusive; causal is `tril(N,M)` (top-left aligned), and a bool mask uses `True=keep`.

## Source-level details

Upstream `decode/tpa_flash_kernel_prefill_mask.py`:

- `tpa_decode_naive_torch` (L970-1037) materializes q/k/v: `torch.einsum("b n h r, b n r d -> b n h d", torch.sigmoid(aq), bq) * scale_q`, keys use `"b m h s, b m d s-> b m h d"`, values use `"b m h s, b m e s-> b m h e"`, then `"b n h d, b m h d -> b h n m"` scores.
- `tpa_decode_torch` (L1039-1106) avoids materialization: `score1 = einsum("b n r d, b m d s -> b n m r s", bq, bk)`, then `score2 = einsum("b n h r, b n m r s -> b n m h s", sigmoid(aq), score1)`, then `score3 = einsum("b n m h s, b m h s -> b h n m", score2, sigmoid(ak))`; softmax multiplies `scale_q * scale_k * scale`; on the output side `prob` is first multiplied by `sigmoid(av)`, contracted with `bv`, and finally scaled by `scale_v`.
- Both paths are mathematically equivalent (sigmoid factors fold into q/k/v); grading only checks the equivalent semantics. The `(B,N,M,R,S)` shape of `score1` is exactly the price of avoiding dense K.
- Masking: `attn_bias = torch.zeros(n, m, dtype=aq.dtype)` has only `(N,M)` axes and broadcasts over batch/heads; with `is_causal` upstream writes `assert attn_mask is None`, which becomes a `ValueError` here.
- A bool mask means `masked_fill_(attn_mask.logical_not(), -inf)`, i.e. `True` keeps; a float mask is added directly (`attn_mask + attn_bias`).
- This file's defaults add two more than the unmasked variant: `scale_k=1/S` and `scale_v=1/S_V` (in `decode/flashtpa_decode_torch.py` they are fixed at 1, there is no sigmoid, and `ak` is the single-rank `(B,M,H)`). The two APIs are not interchangeable; the problem pins the masked one.

## Determinism boundaries

- Everything runs in float64: sigmoid, four scales, softmax, and both output contractions, then casts back to the input dtype; float32 inputs are compared with `3e-5` tolerance.
- `torch.einsum` and numpy may contract in different orders, leaving 1e-16-level differences; tests compare element-wise against an independent numpy implementation rather than requiring bit equality.
- Positions removed by causal or a mask get `exp(-inf)=0`, so `prob` is exactly 0 there and tests assert it.
- An all-`-inf` row (a query with no visible key) makes softmax NaN; the problem excludes such inputs from tests and implementations need no fallback.
- No randomness, no calibration, no trainable parameters; the same input always yields the same output.

## Differences from other problems

- `flash-attention-tile`: tiled online softmax over dense Q/K/V, focused on tiling and rescaling; this problem has no tile loop and focuses on **contraction order and scale placement after factorization**.
- `grouped-query-attention`: several query heads share the same K/V head, reusing the head axis; TPA splits each token's own K/V into two factors and shares the basis `b` across **heads**.
- `kv-xkv-layer-svd-share`: SVD-based cross-layer singular-vector sharing, an offline low-rank compression; TPA is a model-native two-factor parameterization and runs no SVD.
- `causal-mask`: builds only a mask; this problem combines masking, gating, scales, and two contractions.

## Validation

- `check_example`: the 2×3 single-head example in the statement, checking both `prob` rows and `out`, and that causal-masked probabilities are exactly 0.
- `check_matches_oracle`: five shape combinations (including `N != M`, float32/float64, with/without causal) against an independent numpy implementation.
- `check_sigmoid_gate_is_required`: after shifting `aq/ak/av`, a result that skips sigmoid must not match, guarding against plain linear contraction.
- `check_scales_are_required`: each default `1/R`, `1/S`, `1/S_V` takes effect (replacing it with 1 changes the output), and all four explicit scale overrides are verified.
- `check_causal_rectangular`: with `N=3, M=5`, row `i` sees only `j<=i`, rows sum to 1, and row 0 is one-hot.
- `check_bool_and_float_masks_agree`: bool `True=keep` matches the equivalent `0/-inf` additive mask bit for bit, and masked probabilities are 0.
- `check_shapes_and_no_mutation`: output shapes/dtypes are correct, probability rows sum to 1, and none of the six inputs is modified in place.
- `check_is_causal_excludes_mask`: `is_causal=True` together with `attn_mask` must raise `ValueError`.
- `check_validation`: inconsistent R/M/E, mixed dtypes, non-4-D tensors, wrong mask shape/type, illegal scales, and non-bool `is_causal` all raise `ValueError`.
