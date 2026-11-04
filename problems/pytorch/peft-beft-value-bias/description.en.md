# PEFT: BEFT Bypass-Bias Fine-Tuning

## Problem

**BEFT** (Bias-Efficient Fine-Tuning of Language Models in Low-Data Regimes, arXiv:2509.15974, ACL 2026, merged into HF PEFT `tuners/beft`) pushes parameter efficiency to the extreme: no weight matrices are changed and no low-rank factors are added — a value projection gets exactly one **added bypass bias** \(\mathrm{beft\_bias}\ (1, out\_features)\) as its only trainable parameter. The paper's finding: fine-tuning \(b_v\) beats \(b_q\)/\(b_k\) in low-data regimes (a softmax-structure argument: \(b_k\) cancels as a constant shift before the softmax); for bias-free LLMs this is the "add a bias and train it" path.

The deterministic `beft Linear.forward` chain (official `layer.py`):

```
result = base_layer(x)                     # original weights and bias untouched
beft_bias = sum(beft_bias[a].flatten()     # multi-adapter: sum first
                for a in active_adapters)
result = result + beft_bias                # (..., out) + (out,) broadcast
```

With `init_weights=True` the \(\mathrm{beft\_bias}\) starts at all zeros — **training starts at \(\Delta = 0\)**. The merge/unmerge state machine:

```
merge:   base.bias += beft_bias.squeeze()    # ValueError if the base bias is absent
safe_merge: clone-sum then isfinite(...).all(); non-finite values are not written back
unmerge: base.bias -= beft_bias.squeeze()    # restores the original bias
```

The `update_layer` shape account: \(\mathrm{beft\_bias} = \mathrm{randn}((1, out))\), reset to zeros on init; **the per-layer trainable parameter count is exactly out_features** (no weight-matrix increment at all).

## Rules

- `mode="forward_delta"`: given x, weight, beft_bias, base_bias (may be None), return the base linear output plus the bypass bias (broadcast over \((\dots, out)\)).
- `mode="adapter_sum"`: the multi-adapter account — flatten and accumulate each beft_bias; the official forward adds the **sum** to the base output once.
- `mode="merge_bias"`: return base_bias + beft_bias; a bias-free base layer raises ValueError; `safe=True` runs a finite check **on the sum** first.
- `mode="unmerge_bias"`: merged_bias − beft_bias, restoring the original bias exactly (the merge→unmerge round trip is an identity).
- `mode="layer_setup"`: return `{"shape": [1, out], "param_count": out}` plus an all-zero "init" when init_weights; False returns only the shape account (random init has no numeric anchor).
- float64 throughout.

## Example

\(in=4, out=6, x\ (2,3,4)\):

```
forward_delta(x, W, beft, base_bias=b) -> (2,3,6) = x@Wᵀ + b + beft
adapter_sum([beft, beft2]) -> beft.flatten() + beft2.flatten()
merge_bias(b, beft) -> b + beft.squeeze()        # (out,)
unmerge_bias(merged, beft) -> b                  # round trip restores b
layer_setup(6) -> {shape: [1,6], param_count: 6, init: zeros}
```

## Constraints

- The bypass touches neither the original weights nor the original bias: at the zero-init starting point forward equals the base output exactly.
- merge/unmerge is additively invertible: merging several adapters and unmerging them one by one also restores the original bias.
- safe_merge catches two kinds of non-finite values: NaN in an operand and **overflow of two finite values** (the check target is the sum).
- A bias-free layer can still run forward (base = x@Wᵀ), but merge must raise ValueError.
