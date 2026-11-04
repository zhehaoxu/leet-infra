# DeFT decoupled subspace-replacement fine-tuning

DeFT (Decoupled Fine-Tuning) ships in HF PEFT (`peft/tuners/deft/layer.py`). Unlike
LoRA's "low-rank increment", DeFT performs **subspace replacement**: it removes the
weight's projection onto given directions and injects a learnable rank-r correction —
adapted = (I − P_proj)·W + Q_P·R, i.e.

```
delta = Q_P @ (R − rightᵀ @ W)          # full DeFT
delta = −Q_P @ (rightᵀ @ W)             # PaRa (para=True): pure subspace removal
```

where (Q_P, right) come from the projection method and P_proj = Q_P @ rightᵀ.

## Mechanism

### 1. Projection decomposition (fp32 domain)

| method | Q_P | right | P_proj | property |
|---|---|---|---|---|
| `relu` | P | relu(P) | P @ relu(P)ᵀ | non-orthogonal |
| `qr` | Q of qr(P) | Q_P (same matrix) | Q_P @ Q_Pᵀ | orthogonal, idempotent |

All projection math runs in **float32** (QR is unreliable in half precision); callers
cast back.

### 2. Shapes and rank

- P: (out_features, r); R: (r, in_features) (no R in para mode);
- Effective rank capped: `r = min(r, out_features)`; r <= 0 raises ValueError;
- scaling = alpha / r (LoRA-like), 1.0 without alpha.

### 3. Identity initialization

`R_init = rightᵀ @ W / scaling` makes `scaling · R_init == rightᵀ @ W`, so the initial
delta is **exactly zero** — training starts from the pretrained weights and learns the
injection, avoiding the immediate "forgetting" of an initial subspace removal
(`init_weights=False` draws R ~ normal(0, 0.02·init_scale) instead).

### 4. Merge factor caching

`M = rightᵀ @ W` (r × in, roughly r/out_features of the memory) is the only
base-weight-dependent part of the delta; merge caches M and unmerge reconstructs the
exact delta from it (`Q_P @ (R·scaling − M)`). Identity:
`W + delta == (I − P_proj)W + Q_P R`.

### 5. Activation-domain forward

No materialization of W + delta — the update is computed on activations:

```
result = base_product − (base_product @ right) @ Q_Pᵀ + F.linear(x, R·scaling) @ Q_Pᵀ
```

with base_product = x @ Wᵀ (the bias-free base product). **branch returns the full
output** (base_product + increment), with the identity
`out == base_product + x @ deltaᵀ`; under identity init out == base_product (the
adapter is a no-op).

## Judging domain

Single entry `solve(mode, **kw)`, seven modes: `resolve_r` / `resolve_scaling` /
`project` / `merge_factor` / `delta_from_factor` / `init_R` / `branch`. Tolerance
1e-5 on the fp32 pipeline; float64 anchors.
