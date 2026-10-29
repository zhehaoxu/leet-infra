# Deep Dive: GLoRA Weight Merge and Exact Unmerge

## Source

peft `tuners/glora.py` (GLoRA: Generalized LoRA, arXiv:2410.01485). The update splits into an element-wise rescaling of the base weight plus an additive low-rank term, so one adapter has two parallel paths:

- the \(A\) path (`config_A_B`): `lora` is \((d_\text{out},r)@(r,d_\text{in})\), `vector` is a \((d_\text{out},1)\) column, `constant` is a scalar, `empty` is the constant 0;
- the \(B\) path likewise;
- bias side: \(C\) is a \((d_\text{in},1)\) low-rank/column path, \(D,E\) are \((d_\text{out},)\) vectors or scalars.

This problem fixes both \(A\) and \(B\) to the `lora` form and the bias side to vectors — the backbone used by the paper and by PEFT's default configuration.

## Merge formula (`GloraLayer.merge`)

```
new_weight = w0.clone()
for each adapter i:
    new_weight += w0 * A_i + B_i
    new_bias   += b0 * D_i + E_i + W0 @ C_i
```

Three traps:

1. **The accumulator must be separate from `w0`.** Writing `new_weight += new_weight * A + B` would make the second adapter multiply \(W_0+\Delta_1\), creating cross terms. PEFT carries an explicit comment: every adapter's term is computed against the original \(W_0\).
2. **The bias \(C\) term is \(W_0C\) (a matrix product), not \(b_0\odot C\).** \(C\) lives in the input space, so it must pass through the weight before it becomes a \((d_\text{out},)\) vector.
3. **Unmerge must recover \(W_0\) first.** `unmerge` computes \(W_0=(W'-\sum B)/(1+\sum A)\), and only then evaluates \(\sum W_0C\); the \(C\) term uses the **recovered** base weight.

## Why the unmerged forward agrees

In `GloraLinear.forward`, the unmerged path evaluates `F.linear(x, W0*(1+A_i)+B_i, b0*(1+D_i)+E_i+W0@C_i)` per adapter and then does `result -= n * base_result`. Expanded, the weight is exactly \(W_0+\sum_i(W_0A_i+B_i)\) — identical to merge. That is the oracle used by `check_unmerged_forward_matches_merged`, and it is precisely why the `-n*base` line exists.

## Verification approach

- `check_example`: a single-adapter fixture pinning \(W'\), \(b'\) and \(y\) (float32, tolerance \(10^{-5}\)).
- `_dense` expands \(A_dA_u\) as an explicit rank-one sum \(\sum_k d_{\cdot k}\otimes u_{k\cdot}\), structurally different from the reference `@`.
- `_unmerged_forward` takes the per-adapter effective-weight route, independent of the reference's merge-then-forward route.
- `check_zero_init_identity`: PEFT initializes `Xd` to zeros, so \(A=B=C=D=E=0\) and the merged layer is strictly the original one — pinned with `torch.equal`, not `allclose`.
- `check_bias_side_path`: keep only \(C\) and verify \(b'-b_0=W_0C\) with \(W'=W_0\).
- `check_multi_adapter_no_compounding`: match the reference, then assert the buggy rolling update is measurably different, so a sequential implementation cannot slip through.

## Complexity

Merging costs \(O(nd_\text{out}d_\text{in})\) element-wise additions plus \(n\) low-rank products; the forward is a plain `Linear`, with zero adapter overhead at inference time — the whole point of merging.
