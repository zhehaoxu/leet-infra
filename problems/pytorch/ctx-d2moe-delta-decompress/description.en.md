# PEFT: BEFT Bypass-Bias Fine-Tuning

## Problem

**D²-MoE** (Delta Decompression for MoE-based LLMs Compression, arXiv:2502.17298, official repo `lliai/D2MoE`) exploits the high correlation among expert weights for zero-training compression: each expert weight is decomposed into a **shared base weight + expert-specific delta** — the base is a Fisher-weighted merge capturing the shared component, and each delta is compressed by whitened SVD with optionally a V factor shared across experts. This problem pins the deterministic matrix accounts of delta decompression; the semi-dynamic structured pruning of the base (dependent on dynamic probe inputs) is outside the account.

**Fisher merge** (the `merge_experts` fisher branch, independently for w1/w3/w2):

```
coef_j = fisher_j * freq_j / Σ_l (fisher_l * freq_l)
W_base = Σ_j coef_j * W_j
delta_j = W_j - W_base        # a zero-fisher expert has coefficient 0
```

**The parameter-budget rank** (`svd_delta`): \(k = \lfloor r \cdot c \cdot ratio / (r + c) \rfloor\) — \((r{+}c) \cdot k \approx ratio \cdot r \cdot c\) maps a "compressed parameter share"; at ratio=1, \(k = rc/(r{+}c)\) is still < min(r,c), so **no full-rank reconstruction exists**.

**The svdllm whitened SVD with the √Σ split**:

```
W_s = M @ scale;  SVD -> u_k, Σ_k, vt_k
svd_u = u_k @ diag(sqrt(Σ_k))
svd_v = diag(sqrt(Σ_k)) @ vt_k @ inv(scale)
svd_u @ svd_v = (M @ scale)_k @ inv(scale)    # de-whitened truncated reconstruction
```

**The shared V** (`delta_share_V`): the group's deltas are row-stacked into \((m \cdot r, c)\) and factorized once; \(V = \mathrm{diag}(\sqrt{\Sigma_k})\,vt_k\) and each expert keeps the row slice \(U_j = us[j \cdot r:(j{+}1)r, :k]\,\mathrm{diag}(\sqrt{\Sigma_k})\) — satisfying \(U_j V = [\,Us\,\mathrm{diag}(\Sigma_k)\,vts_k\,]_j\) (row slices of the stacked low-rank reconstruction).

**The merged forward** (SwiGLU):

```
gate = x @ (W1_base + U1_j V1).T；up = x @ (W3_base + U3_j V3).T
h = relu(gate) * up；out = h @ (W2_base + U2_j V2).T
```

Routing weights and top-k selection follow the original MoE, multiplied outside the adapter. SVD factors are explicit inputs.

## Rules

- `mode="fisher_merge"`: given weights/fisher/freq, return (base, deltas, coef); coef is normalized and a zero-fisher expert gets coefficient 0.
- `mode="delta_rank"`: given shape and ratio, return \(\lfloor rc \cdot ratio/(r{+}c) \rfloor\).
- `mode="ta_svd_split"`: given whitened-domain SVD factors and inv(scale), return (svd_u, svd_v) split by √Σ.
- `mode="shared_v_split"`: given row-stacked SVD factors, return (V_shared, per-expert U row slices).
- `mode="forward_merged"`: given the three base weights and three delta factor pairs, return the merged SwiGLU forward.
- float64 throughout; no decomposition is run at solve time.

## Example

\(in=6, h=8\), 2 experts, freq=[0.7,0.3], fisher of w1 = [1.5,0.5]:

```
coef = [1.5·0.7, 0.5·0.3] / (1.05+0.15) = [0.875, 0.125]
delta_rank([8,6], 1.0) = int(48/14) = 3        # parameter budget, not full rank
ta_svd_split: svd_u (8,k) @ svd_v (k,6) = (Δ@scale)_k @ inv(scale)
shared_v_split: U_j V = [Us diag(Σ) vts]_j row slices
```

## Constraints

- Whitening cancels: the √Σ-split product equals the whitened-domain truncated-SVD reconstruction right-multiplied by inv(scale).
- The shared-V identity: \(U_j V\) equals, segment by segment, the corresponding row slice of the stacked low-rank reconstruction.
- The merged forward shares one source with an ordinary SwiGLU over the reconstructed (base + delta) weights (cross-checked).
- A zero-fisher expert's delta is still kept (it contributes nothing to the base; its information lives in the delta path).
