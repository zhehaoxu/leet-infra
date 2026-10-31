# Reference approach: where dropout sits, inverted scaling, and gate renormalization

## Implementation

1. Stable softmax: `shifted = logits - logits.max(dim=1, keepdim=True).values`, `scores = exp(shifted) / exp(shifted).sum(1, keepdim=True)`.
2. `expert_idx = torch.sort(scores, dim=1, descending=True, stable=True).indices[:, :k]`; `stable=True` delivers the "smaller index wins ties" rule, whose pure-Python counterpart is `sorted(range(E), key=lambda e: (-scores[e], e))[:k]`.
3. `raw = scores.gather(1, expert_idx)`, `gates = raw / raw.sum(1, keepdim=True)`. It gathers the **softmax scores**, and renormalization happens only across the k picks.
4. `w1[expert_idx]` gives `[T,k,D,H]`; `pre = einsum("td,tkdh->tkh", x, ...) + b1[expert_idx]`; `hidden = relu(pre)`.
5. Dropout:

   ```python
   if training and float(expert_dropout) > 0.0:
       mask = drop_mask.gather(1, expert_idx.unsqueeze(-1).expand(-1, -1, pre.shape[-1]))
       hidden = hidden * mask / (1.0 - float(expert_dropout))
   ```

   `gather` is used instead of `drop_mask[:, expert_idx, :]`, which first broadcasts to `[T,k,E,H]` before selecting — same values, much more memory.
6. `expert_out = einsum("tkh,tkhd->tkd", hidden, w2[expert_idx]) + b2[expert_idx]`, `output = (gates.unsqueeze(-1) * expert_out).sum(dim=1)`.

Common mistakes:

- **Dropping the wrong tensor**: applying dropout to the input `x` or the expert output `y` rather than the expert hidden activation. That is the real semantic boundary here, and the tests catch it on both `hidden` and `output`.
- **Non-inverted scaling** (no training-time scale, multiply by `1-p` at inference). The contract is `1/(1-p)` in training and identity at inference; the other convention is off by exactly `1/(1-p)` on the all-ones mask in `check_dropout_scaling`.
- **Still applying the mask at inference**. With `training=False`, `drop_mask` must be ignored entirely — `check_inference_asymmetry` compares an all-zero mask at `p=0` against the baseline, and sweeps `p∈{0,0.3,0.9}` at inference requiring bit-identical results.
- **Missing the `p=0` short circuit**. At `p=0` with `training=True` the factor `1/(1-p)=1` is harmless, but applying it would also zero the activation when `drop_mask` is all zeros; the "all-zero mask + p=0 + training=True" case in `check_inference_asymmetry` exists for exactly this.

## Verification

Seven `check_*` cover: the worked example (train and inference on the same mask, different outputs), 10 random cross-checks over a grid of `T/E/D/H/k/p/training` including non-contiguous inputs, train/inference asymmetry (inference bit-identical across `p`, and `training=True` with `p=0` unchanged even under an all-zero mask), dropout scaling (all-ones mask scales `hidden` by exactly 2, all-zero mask zeroes it and collapses `output` to `sum_j gates_j * b2[e_j]`), gates (`k=E` reproduces the full softmax, `k=1` gives all ones, a three-way tie picks 0/1/2 by index), degenerate shapes (single token, single expert, width 1), and invalid inputs.

The oracle runs entirely on pure-Python scalars: a hand-written stable softmax via `math.exp`, selection via `sorted(key=lambda e: (-scores[e], e))`, and `math.fsum` for the dot products and weighted sum — no torch `softmax`/`sort`/`einsum`.

Mutation testing: moving dropout onto the input `x`, onto the expert output, dropping the gate renormalization, applying the mask at inference, and using `(1-p)` instead of `1/(1-p)` are all caught. "Missing the `p=0` short circuit" is caught only by `check_inference_asymmetry`, so that case is necessary rather than redundant. Two mutants survive and neither is a coverage gap: replacing stable sort with `torch.topk` agrees on random inputs that contain no exact ties, and the problem constrains the tie outcome rather than the API; and moving dropout **before** relu is numerically equivalent under this contract — positive scaling commutes with relu (`relu(c·a) = c·relu(a)` for `c>0`), for both mask values — so that "position" is not a distinguishable semantic, while the real boundary is that dropout must land on the expert hidden vector.

## Scope and sources

The problem corresponds to Google's ST-MoE (Zoph et al., *ST-MoE: Designing Stable and Transferable Sparse Expert Models*, arXiv:2202.08906). The paper is explicit that "expert dropout" is an independent dropout rate applied to the **expert hidden activation**, separate from global dropout. It also reports trying a variant that masked out entire experts, which did not improve generalisation and was not adopted — so "drop whole experts and renormalize" is not this mechanism's semantics, and the problem does not reproduce it.

In the upstream Mesh TensorFlow implementation (`mesh_tensorflow/transformer/moe.py`), the expert hidden layer is followed by:

```text
h = mtf.dropout(h, is_training=train, keep_prob=1.0 - hparams.moe_dropout_rate)
```

i.e. dropout on the activation before the second projection, gated by `keep_prob`, active in training and off at inference. Mesh's `mtf.dropout` uses inverted scaling by default; this problem writes "multiply by `1/(1-p)` in training, identity at inference" into the contract so it does not depend on a framework default.

Where this problem differs from upstream: the mask is an explicit input rather than RNG; tie order is fixed to "smaller index first" for reproducible judging; and the gate renormalization is presented as the ordinary treatment for sparse top-k gating (not causally tied to dropout) with the contract making it explicit. This problem has no capacity dispatch (see `moe-capacity-jitter-dispatch`), no load statistics (see `moe-dead-expert-ledger`), no gate-bias updates or load-balancing auxiliary loss, and no backward pass.
