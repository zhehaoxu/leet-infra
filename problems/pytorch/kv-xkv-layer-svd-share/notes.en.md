# Solution Notes · xKV Cross-Layer Shared Singular Vectors

## In one sentence

Concatenate the K of every layer in a group along the channel axis, take **one** SVD, share the position-side U across the group, and store only each layer's own `S·V`.

## Why U can be shared

A single layer's key transposed is `[S, H*D]` (rows are positions, columns are channels); concatenating inside the group along `dim=2` gives `[S, gs*H*D]`. Each **row** of it is one position's full description across every layer and head, so the row space found by the SVD is a "position code":

\[
A = U\,\mathrm{diag}(S)\,V^\top,\qquad A_l = U \cdot SV_l,\quad SV_l = (S V^\top)[:,\ l\,H D : (l+1) H D]
\]

Cross-layer redundancy lives exactly in that position code: U is stored once (`num_groups * S * rank`) and `S·V` once per layer (`layers * H * D * rank`) — the compression formula in `scripts/calc_kv_comp_rate.py` upstream is the sum of these two terms.

## Four implementation steps

1. Inside a group: `(gs, S, H, D) → permute(1,0,2,3) → reshape(S, gs*H*D)`, column order "layer → head → dim";
2. `torch.linalg.svd(..., full_matrices=False)`, keep the first `r = min(rank_k, S, gs*H*D)` triples;
3. `SV = diag(S[:r]) @ Vh[:r]`, cut it into `H*D`-column blocks (one per layer), reshape each to `[H, rank, D]`;
4. `recon = U @ SV`, and the residual is the group's Frobenius norm.

## Four conventions that are easy to get wrong

1. **The rank truncates the concatenated width**: the cap is `min(S, gs*H*D)` and may exceed one layer's channel count — that is exactly what makes "scale the rank with the group size" work.
2. **Concatenation order**: upstream's per-layer key is already `[bsz, S, nh*hd]`, so the order is "layer first, head-major inside". Interleaving heads across layers instantly fails the `S=1` case, where `coef` must reproduce the original K.
3. **The trailing group**: the SVD only fires on `layer_idx % gs == gs-1` while `U_k` is allocated with `num_layers // gs` rows, so a non-divisible layer count leaves those layers with neither U nor `S·V`, silently. This problem raises `ValueError` instead.
4. **Determinism**: `torch.svd_lowrank` is randomised and returns different results for the same input (upstream's own comment says "Have deterministic issue but faster"). `linalg.svd` still leaves a sign freedom, so this problem pins "the entry with the largest absolute value of every U column is positive, ties by smallest row index" and flips the matching `S·V` row together, keeping `U @ SV` unchanged.

## What is judged

- the columns of `u` are orthonormal and each obeys the sign convention above;
- `recon` equals `u @ coef` exactly (gauge consistency) and matches an independent numpy truncated reconstruction;
- `err` equals the Eckart–Young tail `sqrt(Σ_{j>r} σ_j²)` and is non-increasing in `rank_k`;
- groups are independent: touching group 0 must not move any output of group 1; two identical layers in a group get element-wise equal `S·V` blocks (the concatenated matrix is `[B|B]`, singular values scaled by `√2`).

## Boundaries

- when `rank_k` exceeds `min(S, gs*H*D)` it is clamped, not an error and not zero-padded; the third axis of `coef` is `r`, not `rank_k`. With a tiny `rank_k` only one direction survives in the whole group and the other layers' coefficients collapse to zero — the inevitable consequence of a shared rank budget (the statement's example is exactly that).
- one `linalg.svd` costs `O(S · (gs*H*D)^2)` and runs once per prefill; the decode-side `u @ sv` gather-gemm is CUDA system code and out of scope here.

## Source

`abdelfattah-lab/xKV` (arXiv 2503.18893):

- `xKV_SR/models/kv_cache_xkv.py:63-79` `fast_svd`: L68 `torch.svd_lowrank(tensor.float(), q=rank)`, L71 `diag_embed(S_trunc) @ Vt_trunc`, L73-74 shape comments for `A`/`B`;
- same file L196 `combined_key = torch.cat(keys, dim=2)`, L211 the in-group `fast_svd`, L215-223 `self.U.append` plus `SVh_trunc.view(bsz, rank, nh*gs, hd).transpose(1,2)` and the per-layer split;
- same file L1243-1249 buffer width `nh*hd*gs` and the slice `start = (layer_idx % gs) * nh * hd`, L1253 `if layer_idx % self.group_size == (self.group_size - 1)`;
- same file L1238-1239 `U_k = zeros(num_layers // gs, ...)` and `SV_k = zeros(num_layers, nh, hd, rank)`: U per group, SV per layer;
- same file L1263-1264 use of `rank_k`/`rank_v` for K/V, L1271-1280 the per-layer split (the K path stores `[bsz, nh, hd, rank]`, the V path `[bsz, nh, rank, hd]`; this problem uses the latter);
- `README.md:99-100`: `--layer_group_size` defaults to 1 with 2/4 for xKV; `--rank_k/--rank_v` default to 256/768 and scale with the group width; `scripts/calc_kv_comp_rate.py:24-27` gives the memory accounting;
- `efficiency/tests/test_svd_reconstruction_accuracy.py:36-39`: uses `sqrt(Σ tail σ²)` as the reconstruction yardstick (this problem's `err`).

**Additions of this problem** (not upstream, pinned explicitly): exact `torch.linalg.svd` instead of the randomised `svd_lowrank`, the sign convention and its tie rule, a hard error on a non-divisible layer count, `err` as a return key, and the `[layer][head][dim]` flattening order.

## Related problems

- `kv-palu-rank-budget`: also allocates rank per layer group, but judges how the budget is split between groups; this problem judges the single in-group SVD and how it is shared.
- `kv-lococo-conv-recombine`: shares one merged K/V across layers, while this problem shares singular vectors and still stores per-layer coefficients; `kv-wkvd-least-squares-merge` solves a least-squares problem, this one only truncates and splits.
