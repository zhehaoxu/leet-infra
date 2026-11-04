# Weighted adapter merging (add_weighted_adapter)

PEFT's `LoraModel.add_weighted_adapter` (`peft/tuners/lora/model.py`) combines several
trained LoRA adapters into a new one with per-adapter weights — the **offline merging
ledger** (task arithmetic, sign splitting, SVD rank reduction in one pass) that runs
before deployment. Delimited from `peft-lora-merge` (single-adapter fold W+sBA) and
`peft-ties-dare-merge` (paper-level TIES/DARE operators): this problem tests the
**library mechanism itself** — combination-type dispatch, rank rules, scaling
normalization, and the exact formulas of the three merge paths.

## Mechanism

### 1. Rank rules (`_check_add_weighted_adapter`)

- **A single source forces linear**: with one adapter, the requested combination_type
  is replaced by `linear`;
- **The linear family** (linear/ties/dare_ties/dare_linear/magnitude_prune): all source
  ranks must be equal (ValueError otherwise), new_rank = that value;
- **cat**: new_rank = sum of the source ranks (may OOM, as the docs warn);
- **\*_svd family**: new_rank = `svd_rank or max(ranks)`;
- any other combination_type → ValueError.
- Each adapter's participating rank is `config.r`, or `max(r, *rank_pattern.values())`
  when rank_pattern is non-empty (allocation upper bound).

### 2. Scaling normalization

The new adapter's config is forced to `r = lora_alpha = new_rank`, `use_rslora=False`,
`alpha_pattern/rank_pattern={}` — i.e. **the new adapter's scaling is exactly 1** — so
each source's scaling must be folded into the merge weights (otherwise it would be
counted twice). Inheriting use_rslora=True would make the scaling r/√r = √r and break
the invariant.

### 3. cat branch (exact)

- lora_A is multiplied by `weight * scaling[adapter]`, lora_B is **unscaled**;
- A is concatenated along dim 0 (the rank dim), B along dim 1;
- afterwards `A_new @ B_new = Σᵢ wᵢ·sᵢ·BᵢAᵢ` holds exactly.

### 4. linear branch (rough approximation; sqrt split, one-sided sign)

`weight_with_scaling = weight * scaling[adapter]`, take `sqrt(|·|)`, and put the
**sign on the A side only**:

```
w_A = sqrt(|w·s|) · sign(w·s)，w_B = sqrt(|w·s|)
A_new = Σ w_A·Aᵢ，B_new = Σ w_B·Bᵢ
```

With two sources the cross terms make `A_new @ B_new ≠ Σ wᵢsᵢ·BᵢAᵢ` (the docs call
linear a rough approximation); with a single source it is exact. If both sides carried
the sign, negative weights would cancel — hence the one-sided sign.

### 5. svd branch (exact; kohya convention)

- First build each full delta via `get_delta_weight`: `deltaᵢ = (Bᵢ @ Aᵢ) · sᵢ`
  (**scaling already folded in**);
- combine with the **raw weights**: `combined = Σ wᵢ · deltaᵢ` (negative weights act
  naturally);
- after the SVD, **the singular values are absorbed into B**: `B = U[:, :rank] @
  diag(S[:rank])`, `A = Vh[:rank, :]` (the kohya-ss sd-scripts convention — not a
  sqrt split);
- `svd_clamp`: take the quantile hi of `cat([U.flatten(), Vh.flatten()])` and clamp
  U/Vh symmetrically to [−hi, hi];
- the return order is `(A, B) = (Vh, U)`.

### 6. Bias merging

`new_bias = Σ wᵢ · sᵢ · bᵢ`, accumulating only sources with lora_bias (scaling
multiplied back explicitly); all-None raises ValueError.

## Judging domain

Single entry `solve(mode, **kw)`, seven modes: `effective_rank` / `resolve_rank` /
`resolve_scaling` / `merge_cat` / `merge_linear` / `merge_svd` / `merge_bias`. Float64
numeric anchors; the Embedding/Conv branches and the TIES/DARE families are out of
scope (TIES/DARE live in `peft-ties-dare-merge`).
