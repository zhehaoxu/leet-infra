# OSF orthogonal subspace fine-tuning

OSF (Orthogonal Subspace Fine-tuning) ships in HF PEFT (`peft/tuners/osf/`), designed
for **continual learning**: an adaptive SVD splits each weight into a frozen
high-rank part (preserving important directions from previous tasks) and a trainable
low-rank part, and training gradients are **projected to be orthogonal to the
high-rank subspace** — full fine-tuning expressivity without catastrophic forgetting
and without adding parameters.

## Mechanism

### 1. Weight decomposition

```
W ≈ U_high · diag(S_high) · V_high + U_low · diag(S_low) · V_low
```

- The SVD runs in **float32** (`W.to(torch.float32)`, `full_matrices=False`) and the
  results are cast back — identity tolerances must follow the fp32 pipeline (1e-5);
- `k = min(top_k, min_dim)`: the first k singular components freeze (high), the rest
  train (low);
- `rank_high = k`; out-of-range `top_k` clamps to min_dim (empty low part).

### 2. Rank resolution (effective_rank semantics)

**`effective_rank` is the PRESERVED rank, not the trainable rank** — trainable =
min_dim − effective_rank, the opposite of LoRA's r:

| input | resolution |
|---|---|
| int | preserved rank as-is |
| float in (0, 1] | `int(fraction · min_dim)` |
| None | `min_dim // 2` (default 50%) |

Out-of-range floats (<= 0 or > 1) raise ValueError.

### 3. Gradient orthogonal projection

```
dU ← dU − U_high @ (U_highᵀ @ dU)      # orthogonalize against U_high's column space
dV ← dV − dV @ (V_highᵀ @ V_high)      # orthogonalize against V_high's rows (right Gram)
```

Judge-sensitive points:

- **The U projection left-multiplies a projector; the V projection right-multiplies
  a Gram matrix** — the two formulas are asymmetric;
- Orthogonality holds only for **orthonormal U_high columns / V_high rows** (true of
  SVD factors) — tests must build them with QR; random Gaussians break the
  orthogonality assertion;
- The minus sign: the plus mutant turns "remove the high component" into "double it".

Distributed all-reduce branches are out of scope.

### 4. Reconstruction

`U_high diag(S_high) V_high + U_low diag(S_low) V_low` recovers W exactly within the
fp32 pipeline; the high part alone is the best rank-k approximation.

## Judging domain

Single entry `solve(mode, **kw)`, five modes: `decompose` / `reconstruct` /
`resolve_rank` / `project_u` / `project_v`. The SVD input W is given by the test
cases (deterministic); float64 anchors with 1e-5 tolerance on the fp32 pipeline.
