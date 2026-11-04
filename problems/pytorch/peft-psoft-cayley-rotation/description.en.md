# PSOFT Cayley orthogonal rotation between SVD factors

PSOFT ships in HF PEFT (`peft/tuners/psoft/layer.py`). It SVD-decomposes the base
weight W and caches the A/B factors, placing a **learnable r×r orthogonal rotation R**
between them: ΔW = scaling · B(R − I)A. Initialization sets R = I (all
upper-triangular params zero → Q = 0 → Cayley identity), so the initial delta is zero.

## Mechanism

### 1. Skew-symmetric parametrization

With `orth=True` only `size(size−1)/2` upper-triangular params are stored;
`index_put` into the upper triangle then `Q − Qᵀ` yields a skew-symmetric Q
(zero diagonal, Q[rows[k], cols[k]] = w[k]).

### 2. Cayley transform

```
R = (I + Q)(I − Q)⁻¹
```

The official code uses `torch.linalg.solve(I − Q, I + Q, left=False)` — `left=False`
solves `X·A = B`, i.e. `X = B·A⁻¹`. Q commutes with its own rational function, so
`left=True` ((I−Q)⁻¹(I+Q)) is numerically equivalent. R is a proper rotation
(det=+1) and orthogonal (R·Rᵀ = I). Zero params ⇒ R = I.

**Cayley-Neumann series** (solver-free):

```
R = I + 2Q + 2Q² + 2Q³ + … + 2Q^(t−2) + Q^(t−1)
```

Coefficient rule: terms 1..t−2 carry coefficient 2.0, the **final term carries 1.0**
(the truncated series is generally not orthogonal). Optional Frobenius projection:
when `norm(Q) > eps`, `Q ← Q · eps/(norm + 1e-12)` (the +1e-12 is numerically
invisible under the norm > 0.9 constraint — a true-equivalence domain).

### 3. Magnitude vectors

```
R = diag(vector_b) @ R @ diag(vector_a)    # row scaling vector_b[:, None] * R, column scaling R * vector_a[None, :]
```

Both vectors initialize to 1 (identity).

### 4. SVD factor cache and A/B init

Full mode `torch.linalg.svd(W)` (deterministic for a fixed input — legitimate in the
judging domain): `Vr = U[:, :r]`, `Sr = S[:r]`, `Uhr = Vh[:r, :]`.

- `psoft_init`: A = Uhr (raw right singular vectors), B = Vr·diag(Sr/scaling)
  (all singular values go into B);
- `pissa_init`: A = diag(√(Sr/scaling))·Uhr, B = Vr·diag(√(Sr/scaling)) (sqrt split).

Both inits give the same B@A product (== truncated W / scaling). Unknown init raises
ValueError.

### 5. Delta and forward

```
ΔW = scaling · B (R − I) A          # R = I ⇒ ΔW = 0
xa = x @ Aᵀ;  xr = xa @ Rᵀ
branch = (xr − xa) @ Bᵀ · scaling    # equals x @ ΔWᵀ
```

r validation: `r > min(in, out)` raises ValueError.

## Judging domain

Single entry `solve(mode, **kw)`, eight modes: `skew_symmetric` / `orth_matrix` /
`magnify` / `svd_factors` / `init_ab` / `delta` / `branch` / `r_check`. The SVD input
W is given by the test cases (deterministic full mode), which resolves the earlier
judging-domain concern; numeric anchors live in float64.
