# Notes: AdaMSS Adaptive Multi-Subspace Scatter Adaptation

## Provenance

The AdaMSS paper (NeurIPS 2025, Zheng et al., official repo `jzheng20/NeurIPS2025-AdaMSS`, merged into HF PEFT `tuners/adamss`). Motivation: the orthogonal fine-tuning family (BoFT/OFT/diagonal-FFT lineage) confines \(\Delta W\) to a single low-rank orthogonal subspace and treats all singular directions alike. AdaMSS observes that SVD spectral components differ in importance — so the \(r\) components are clustered into groups, each with its own low-rank factors and its own rank (possibly frozen entirely), adaptively allocating capacity along the spectrum within a fixed parameter budget. The frozen \(\mathrm{newB} = V_r'^\top\) guarantees \(\Delta W\) always lives in the direct sum of the preselected subspaces.

## Line-by-line translation of the official implementation

The deterministic trunk of `layer.py` (training/device details removed):

```python
# end of update_layer: scatter index = per trained subspace, concatenated
self.scatter_index[adapter_name] = torch.cat(
    [self.seg_result[adapter_name][self.train_subspace_index[adapter_name][i]]
     for i in range(self.num_subspaces[adapter_name])], dim=0).long()

# forward: ones column appended when bias exists (output dim not incremented)
if self.get_base_layer().bias is not None:
    ones = torch.ones(*x.shape[:-1], 1, ...)
    newx = torch.cat((x, ones), dim=-1)
projected = F.linear(newx, self.adamss_newB[...])          # (..., r)
for i in trained_subspaces:
    h = projected @ A_i.T
    outs.append(h @ B_i.T)                                  # (..., seg_i)
delta = torch.zeros(..., out_features).scatter(1, index, cat(outs))

# merge trunk (shared by get_delta_weight / get_delta_bias):
blocks = [B_i @ A_i for ...]
combined = torch.cat(blocks, dim=0) @ self.adamss_newB[...]  # (Σ seg_i, in+1)
delta_weight = torch.zeros(out_features, in_features_plus_1)
delta_weight[scatter_idx, :] = combined
# get_delta_weight: output_tensor = delta_weight[:, :-1]
# get_delta_bias:   output_tensor = delta_weight[:, -1]
```

Three key readings:

1. **seg partitions the out-dim SVD components** (covering \(0..r{-}1\)), not the input dim; \(r \le out\), and output rows without a component keep a zero delta.
2. **The rank clamp has two guards** (fixed-rank mode): `actual_rank = min(len(seg_indices), subspace_rank)`, forced to 1 when zero — empty segments and subspace_rank=0 never produce a zero-rank subspace. (The dynamic-rank mode adds a Gram-matrix SVD threshold filter, outside this problem's account.)
3. **Bias is a column, not a row**: bias enters the SVD space via a ones column on the input side (the last column of \(W'\) multiplies 1), so the merged \((out, in{+}1)\) augmented delta splits along **columns**; `get_delta_bias` shares the entire trunk with `get_delta_weight` and differs only in the final strip.

**Forward and merge are the same source**: forward is the stepwise form of \(\mathrm{cat}[\,(B_iA_i)\,]\cdot \mathrm{newB}^\top\) (project first, then lift per segment), while merge lifts first, multiplies \(\mathrm{newB}\) once, then scatters — cross-checked via \(x\, @\, W_\Delta^\top\ (+b_\Delta)\).

## Pitfalls

1. **The output dim does not get +1**: bias is folded into the input-side ones column, so forward still outputs out_features; adding the +1 on the output side (the double-plus-one bug) is the classic shape error pinned here (m-fwd-outp1).
2. **Scatter order = trained-subspace order**: `build_scatter_index` concatenates in train_idx order; reversing or ignoring train_idx misplaces rows (m-scatter-rev / m-scatter-all), and resolving segments by position instead of group id also breaks (m-scatter-pos).
3. **The rank clamp has two bounds**: dropping the floor (empty segment -> 0), dropping the cap (beyond subspace_rank), or raising the floor to 2 are all observable (m-rank-nofloor / m-rank-nocap / m-rank-floor2).
4. **Each merge block is multiplied by newB individually**: `cat([B_i@A_i]) @ newB` differs numerically from dropping the multiply or swapping the A/B factors (m-dw-nonewb / m-dw-ba); reversing the blocks misplaces scattered rows (m-dw-catrev).
5. **Rows are scattered, not left-aligned**: `dW_full[scatter_index, :] = combined`; left-aligned writes misplace rows immediately on the partial-training fixture (scatter=[0,1,4]) (m-dw-rowassign / m-fwd-noscatter).
6. **The split is column-wise**: bias takes `[:, -1]`, not `[-1, :]` (m-split-row); the weight strips `[:, :-1]`, not `[:, 1:]` (m-split-off); short-circuiting the with_bias branch is caught by the forward_bias cross-check (m-split-nobias).
7. **The with_bias projection is dimension-coupled**: newB \((r, in{+}1)\) only pairs with \([x\,|\,1]\); a no-bias input against a bias-carrying newB is a shape violation (the tests pin the ValueError guard).
8. **Untrained rows stay zero**: with partial training, forward and merge are zero on untrained rows — the fixture deliberately skips a middle group to pin scatter semantics (forward_partial / dW_partial anchors).

## Cross-check account

Anchors dumped from the sandbox (numpy float64, seed 31): out=6/in=5/r=5, three groups (2,2,1), subspace_rank=2. Three-way cross-checks — (1) no-bias: forward \(= x\, @\, \mathrm{dW}^\top\); (2) with-bias: \(x\, @\, W^\top + b =\) forward, plus the equivalent form \(x_{aug}\, @\, \mathrm{dW}_{full}^\top\); (3) partial training: zero assertions on both untrained rows and columns. rank_clamp over four configurations (sr=1/2/5 plus an empty segment). All 23 mutants CAUGHT, no equivalent mutations.

## Complexity

Forward costs \(O(BN\cdot r + \Sigma_i\,\mathrm{rank}_i\cdot \mathrm{seg}_i)\), dominated by the frozen projection \(x_{aug}\ @\ \mathrm{newB}^\top\); merge adds one \((\Sigma \mathrm{seg}_i, r)\ @\ (r, in{+}1)\). Trainable parameters \(\Sigma_i\,\mathrm{rank}_i(\mathrm{seg}_i + r) < r\cdot(out+r)\) — cheaper than training the whole left factor UU, with capacity allocated along the spectrum.
