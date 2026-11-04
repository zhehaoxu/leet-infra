# Notes: D2-MoE Delta Decompression

## Provenance

The D²-MoE paper (arXiv:2502.17298, Hao Gu et al., HKUST + Birmingham + AISpeech, official repo `lliai/D2MoE`). Motivation: MoE LLM expert weights are highly correlated — representing each expert as "shared base + exclusive delta" stores the base once and compresses deltas at low rank, reducing storage from \(m \cdot out \cdot in\) to \((out \cdot in) + m \cdot k(r{+}c)\). The three-stage method: (1) Fisher-weighted merge produces the base; (2) deltas are compressed by whitened SVD with optional cross-expert V sharing; (3) semi-dynamic structured pruning of the base (not in this problem — its scoring depends on dynamic probe inputs, implemented in pruning_module.py). Beats baselines by 13%+ at 40~60% compression on Mixtral/Phi-3.5/DeepSeek/Qwen2-MoE.

## Line-by-line translation of the official implementation

```python
# merge_mixtral.py: merge_experts, fisher branch (w1 example)
fisher_scale_w1[j] = hessian[key_w1] * scaling_factors_ft_list[j]   # fisher × expert_freq
fisher_scale_w1 = [f / sum(fisher_scale_w1) for f in fisher_scale_w1]
self.Wmean1.weight.data = sum(experts[j].w1.weight * fisher_scale_w1[j] for j in range(8))
# delta: module.experts[j].w1.weight - self.Wmean1.weight

# svd_delta: parameter-budget rank + svdllm whitening
num_s_after_trunc = int(W.shape[0]*W.shape[1]*ratio / (W.shape[0]+W.shape[1]))
W_scale = W @ svd_scale                      # right-multiply whitening
U, S, VT = torch.linalg.svd(W_scale)
truc_v = VT[:k] @ cal_scale_inv(svd_scale)   # de-whitening
svd_u = U @ sqrt(Σ); svd_v = sqrt(Σ) @ truc_v  # sqrt(Σ) split on both sides

# delta_share_V: row-stacked shared V
# stack (8*d, n), one SVD -> shared V, per-expert U row slices;
# shared U averages by freq; shared scale averages the scales by freq
```

Three key readings:

1. **The fisher×freq double weighting**: the merge weight is the normalized product of the Fisher information and the expert routing frequency — frequent and important experts dominate the base; a zero-fisher expert has coefficient 0 (nothing in the base, its information fully kept in its delta, which is never dropped).
2. **The parameter-budget rank is never full rank**: \(k = \lfloor rc \cdot ratio/(r+c) \rfloor\) maps a "compressed parameter share" to a rank — at ratio=1, \(k = rc/(r+c) < \min(r,c)\), so every ratio is lossy and there is no "full-rank reconstruction" anchor (cross-checks go through the whitened-domain truncated-SVD reconstruction identity instead).
3. **The √Σ two-sided split with de-whitening**: `svd_u = U√Σ`, `svd_v = √Σ·Vt·scale⁻¹`; the product = \((M \cdot scale)_k \cdot scale^{-1}\) — whitening only changes the SVD metric and reconstruction right-multiplies inv(scale) to return to the original domain. With a shared V, the V side carries √Σ and the U row slices also carry √Σ, so \(U_j V\) is exactly the row slice of the stacked low-rank reconstruction \(Us\,\mathrm{diag}(\Sigma)\,vts\).

**The merged forward** (shared_infer): the base path runs once for all tokens, the shared V once as well; output = base SwiGLU + delta low-rank SwiGLU added in place; routing softmax/top-k/weighting stay in the MoE layer unchanged.

## Pitfalls

1. **The double weighting is inseparable**: dropping freq (m-merge-nofreq) or the normalization (m-merge-nonorm) changes the base and every delta; dividing instead of multiplying (m-merge-swap).
2. **The delta direction**: \(W - base\), not \(base - W\) (m-delta-rev — a flipped direction breaks the forward numerically immediately).
3. **Two rank traps**: int truncation, not round (m-rank-round); the numerator is \(r \cdot c \cdot ratio\) and the denominator \(r+c\) — transposing them yields 0 (m-rank-flip).
4. **The √Σ split is two-sided**: one side only under-multiplies by Σ (m-svd-oneside); dropping de-whitening (m-svd-noinv) or multiplying scale twice (m-svd-scale).
5. **Three parts of the shared V**: V carries √Σ (m-shv-nosq); the U-slice start is \(j \cdot r\) (m-shv-offset — invisible at j=0, so pin the j≥1 segment); the U slices carry √Σ too (m-shv-unosq).
6. **The SwiGLU structure**: ReLU applies to gate, not up (m-fwd-actswap); the two channels multiply elementwise, not add (m-fwd-add); the delta low-rank path adds in place with the base and dropping either path is wrong (m-fwd-nodelta).
7. **Equivalent-mutation record**: re-int-ing an already-int return in dispatch is an identity (m-disp-rank SURVIVED, recorded and removed).

## Cross-check account

Anchors dumped from the sandbox (numpy float64, seed 808): in=6/h=8, 2 experts, freq=[0.7,0.3], per-matrix fisher for w1/w3/w2. Four cross-check lines — (1) whitening cancellation: the √Σ-split product ≡ \((M \cdot scale)_k \cdot scale^{-1}\); (2) the shared-V identity: \(U_j V\) ≡ row slices of the stacked \(Us\,\mathrm{diag}(\Sigma)\,vts\) low-rank reconstruction; (3) the merged forward ≡ an ordinary SwiGLU over the reconstructed base+delta weights; (4) with zero deltas, forward ≡ the pure base SwiGLU. The rank account over 8 configurations (two shapes × ratio 0.2/0.5/0.9/1.0). Of 16 mutants 15 CAUGHT plus 1 int-identity equivalent.

## Complexity

The merge is offline and costs \(O(m \cdot out \cdot in)\) once; compressed storage \(out \cdot in + m \cdot k(r{+}c)\), saving about \(m(1-ratio) \cdot out \cdot in\) at delta_ratio=0.8; the shared V saves another \((m{-}1) \cdot k \cdot c\). The forward's base path is shared across all tokens and the delta low-rank path \(O(BN(rk + kc))\) only runs for selected experts.
