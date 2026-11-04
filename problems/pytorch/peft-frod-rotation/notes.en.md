# Notes: FRoD Rotation with Diagonal plus Sparse Delta

## Source

FRoD (Frequency-domain Rotational Decomposition, AAAI 2026) landed in HF PEFT v0.20.0 (`src/peft/tuners/frod/`). Design motivation: deltas of same-class layers (e.g. all q_proj) are highly correlated, so per-layer LoRA wastes parameters — FRoD shares one **rotation basis** \(V\) across layers and expresses each layer's delta as "diagonal + extremely sparse off-diagonal". The diagonal term is a rank-1 scaling view (one coefficient per basis direction); the sparse term \(L\) (density controlled by `sparse_rate`) supplies off-diagonal corrections.

## Line-by-line translation of the official implementation

**Shared projection (`model.py::get_frod_projection`)**:

```python
w_cat = torch.cat([w for w in same_class_weights], dim=0)   # (Σout, in)
if w_cat.shape[0] < w_cat.shape[1]:        # thin: rows < cols
    _, _, vh = torch.linalg.svd(w_cat.float(), full_matrices=False)
    V = vh                                  # row-orthogonal
else:                                       # wide: rows ≥ cols
    q, r = torch.linalg.qr(w_cat.float().T, mode="reduced")   # q:(in,k), r:(k,Σout)
    invr = torch.linalg.inv(r.T @ r + alpha * torch.eye(...)) # ridge average
    m = (invr @ r.T @ q.T).mean(...)        # symmetrize → eigh
    eigval, eigvec = torch.linalg.eigh(m)
    V = eigvec                              # square (in, in)
```

**Sparse COO (`model.py`)**:

```python
all_idx = [(i, j) for i in range(n) for j in range(n) if i != j]  # minus diagonal
nnz = max(1, int(n * n * sparse_rate))
perm = torch.randperm(len(all_idx), generator=torch.Generator().manual_seed(prng_key))
s_indices = all_idx[perm[:nnz]]            # frod_s_indices, values zero-init
```

**Per-layer bookkeeping (`layer.py::reset_frod_parameters` + `get_delta_weight`)**:

```python
wv = W @ V.T                               # (out, in)
if wv.shape[0] == wv.shape[1]:
    u = torch.linalg.solve(V, W.T).T       # official solve(V, Wᵀ)ᵀ: U diag(λ) = W Vᵀ
else:
    u = (torch.linalg.pinv(V) @ W.T).T
lambda_l = wv.norm(dim=0)                  # column norms (in,)
u = u / u.norm(dim=0, keepdim=True)        # column-normalize
s_values = zeros(nnz)
# get_delta_weight:
S = scatter(s_values, s_indices, (in, in))
delta_w = u @ (S + diag(lambda_l)) @ V.T - W   # pre-correction minus W
```

**Forward (`layer.py::forward` + `_sparse_activation_mm`)**:

```python
z = x @ V
z_S = _sparse_activation_mm(z, S)          # (S @ z.T).T = z @ S.T   ← the asymmetric-S transpose subtlety!
z_L = z * lambda_l
out = base(x) + (z_S + z_L) @ u.T
```

## Three numerical conventions (pitfalls)

1. **\(S\) is not symmetric; forward is \(zS^{\top}\)**: `_sparse_activation_mm` is implemented as `S @ z.T` transposed back, i.e. `z @ S.T`. Writing `z @ S` naively is transpose-inconsistent with merge's \(U(S+L)V^{\top}\) (the sandbox cross-check catches it immediately; pinned by m-fwd-notrans). The diagonal branch \(z*\lambda\) has no transpose issue (diagonal matrices are symmetric).
2. **When \(S=0\), \(\Delta W\equiv 0\) exactly**: \(U\mathrm{diag}(\lambda)V^{\top}=W\) is a constructive identity (\(U\lambda\) is a rearranged polar factorization of \(WV^{\top}\)). This is the "zero init = zero delta" semantics of the factorization-domain parameterization, same philosophy as LoRA's zero-B init. Note it no longer holds when \(L\ne0\) — \(L\) is the post-training sparse residual.
3. **solve vs pinv**: the square path officially uses `torch.linalg.solve`; for non-square W (out≠in) the non-invertible path falls back to `pinv`. With an ill-conditioned \(V\) (large condition number), pinv's rtol truncates small singular directions and the results are numerically distinguishable — the tests pin the solve path with a cond≈\(10^8\) fixture.

**The λ/U normalization account**: \(\lambda\) takes the column norms of \(WV^{\top}\) and \(U\) is column-normalized, keeping \(U\mathrm{diag}(\lambda)=WV^{\top}\) exact; during training \(\lambda\) (diagonal) and `s_values` (sparse) are trainable while \(U,V\) stay frozen.

**float32 boundary**: the factorization of \(V\) and sparse-index generation run in float32 (official path); W / x / ΔW external bookkeeping is float64. Test anchors are dumped along the official dtype path (V float32 values reproduced bit-for-bit).

## Cross-check conventions

Anchors dumped from the sandbox: F1 wide path (two layers concatenated, 23 rows > 6 cols, QR+ridge+eigh), F2 thin path (rows < cols, SVD row-orthogonal), F3 sparse COO (nnz account + off-diagonal + seed determinism), F4 per-layer solve/λ/normalization + reconstruction identity, F5 zero-S identity ΔW≡0, F6 non-square pinv fallback. Cross-checks: \(x\Delta W^{\top} = xV(S+L)^{\top}U^{\top}\), and exact zero when S=0.

## Complexity

The shared projection is a one-shot \(O((\sum out)\cdot in^2)\); each layer's forward is \(O(BN(in^2))\) (dominated by the square \(V\) multiply) — slightly more than a same-rank LoRA (\(V\) is \(in\times in\) rather than \(in\times r\)), but the trainable parameters are only \(in + nnz\) diagonal + sparse slots, an order of magnitude below \(r(in+out)\).
