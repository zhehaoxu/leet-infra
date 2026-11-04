# PEFT: BD-LoRA Block-Diagonal Factors for Tensor-Parallel Serving

## Problem Description

**BD-LoRA** (arXiv:2510.23346, NeurIPS 2025, Amazon, merged into HF PEFT as `use_bdlora`/`BdLoraConfig`) constrains some LoRA factors to be **block-diagonal** so that tensor-parallel serving becomes **communication-free**: a column-parallel layer requires \(B\) to be block-diagonal and a row-parallel layer requires \(A\) to be block-diagonal, with each block-diagonal shard resident on one GPU, eliminating all-gather / all-reduce. Equivalent interpretation: BD-LoRA attaches an independent rank-\(r/N\) LoRA to every shard of the base weights. The block-diagonal constraint reduces expressiveness, compensated by increasing \(r\) for parameter parity.

This problem replicates the deterministic account of the official PEFT implementation (`BlockDiagonalLinear` + `BdLoraLinearVariant`):

- **Compact storage and forward**: the factor is not stored as an \((out, in)\) full matrix but as `nblocks` stacked shards of shape `(out_features, in_features // nblocks)`; the forward maps `x -> (B, nb, m)`, `weight.view(nb, n, m)`, multiplies per shard with `einsum("bim,inm->bin")` and concatenates, keeping leading dims for 3-D input. Non-divisible shapes raise ValueError.
- **Block-diagonal expansion**: `torch.block_diag(*torch.chunk(weight, nblocks, dim=0))` restores the full layout with strictly zero off-diagonal blocks.
- **Target classification**: substring match against `target_modules_bd_a` / `target_modules_bd_b`; matching BOTH groups raises ValueError; under `match_strict=True` matching NEITHER raises ValueError, while `False` falls back to plain LoRA.
- **Init**: the block-diagonal `lora_A` uses `kaiming_uniform_(a=√5)` (fan_in read from the shard's \(in/nb\)), and `lora_B` is all-zero (so \(\Delta W=0\) at start).
- **Merge account**: on the expanded matrices `delta = (B_bd @ A_bd) · scaling` (linear layers, `fan_in_fan_out=False`, no transpose); the merged base weight reproduces the unmerged two-step forward **exactly**.

## Task

- `mode="bd_forward"`: inputs x, weight (compact shards), nblocks; return the block-diagonal forward output; ValueError when not divisible.
- `mode="block_diag_matrix"`: inputs weight, nblocks; return the expanded full matrix.
- `mode="classify_target"`: inputs target_name, patterns_a, patterns_b, match_strict; return "a"/"b"/"plain"; invalid combinations raise ValueError.
- `mode="init_factors"`: inputs in_features, out_features, r, nblocks, role ("a"/"b"), seed; return the compact weight (A kaiming anchored by seed, B zeros); invalid role raises ValueError.
- `mode="delta_weight"`: inputs weight_a, weight_b, nblocks, scaling, fan_in_fan_out; return the \((out,in)\) delta.
- `mode="layer_forward"`: inputs x, base_weight, base_bias, weight_a, weight_b, nblocks, scaling; return `base + lora_B(lora_A(x))·scaling`.

## Example

seeds 0/1/2/3, float64:

```
bd_forward(x (4,8), weight (6,4), nblocks=2) -> y (4,6)
  y[0,0] = -1.2027563104668666, y[-1,-1] = 2.3503342488855252
block_diag_matrix(weight (6,4), 2) -> bd (6,8), bd[0,0] = -0.31128955419003207
  bd_forward(x) == x @ bd^T; off-diagonal blocks strictly zero
classify_target("o_proj", ["o_proj"], ["q_proj"], True) == "a"
init_factors(8, 6, 4, 2, "a", seed=0) -> (4,4), wa[0,0] = 0.470053001806553
init_factors(8, 6, 4, 2, "b") -> (6,2) all zero
delta_weight(wa, wb (6,2), 2, scaling=2) -> (6,8), delta[0,0] = -0.44661549644151793
layer_forward(...) == F.linear(x, base_w + delta, base_b) (merge identity)
```

## Constraints

- `in_features`, `out_features` (and the r-side dims) must be divisible by `nblocks`, else ValueError.
- The expanded block-diagonal matrix has **strictly zero** off-diagonal blocks (`abs().sum() == 0`) and diagonal blocks `torch.equal` to the shards.
- `match_strict=True` with no matching pattern -> ValueError; matching both A and B patterns -> ValueError.
- `fan_in_fan_out=True` transposes the delta (Conv1D-style weights).
- Merge identity: `layer_forward(x) == F.linear(x, base_w + delta, base_b)`.
