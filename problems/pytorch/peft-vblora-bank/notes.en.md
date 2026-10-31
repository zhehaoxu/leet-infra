# Notes · VB-LoRA's Vector-Bank Selection

## Where it comes from

- `huggingface/peft`, `src/peft/tuners/vblora/layer.py`: `_get_low_rank_matrix(logits, bank, topk)` does `logits.topk(topk, dim=-1)`, then `F.softmax(top_k_logits, dim=-1)`, then `(weights.unsqueeze(-1) * bank[indices]).sum(-2)`; `_get_lora_matrices` flattens the A side with `reshape(r, -1)` and the B side with `transpose(1, 2).reshape(-1, r)`; `get_delta_weight` returns `transpose(B @ A, fan_in_fan_out)`; the unmerged path of `Linear.forward` is `result + F.linear(F.linear(dropout(x), A), B)`. This problem mirrors those four fragments line by line.
- The same file's `update_layer` supplies the validations: `r > 0`, `topk > 0`, `in_features % vector_length == 0`, `out_features % vector_length == 0`, and fixes `vblora_logits_A` to `(r, in//vl, V)` and `vblora_logits_B` to `(out//vl, r, V)`.
- Paper: Li et al., *VB-LoRA: Extreme Parameter Efficient Fine-Tuning with Vector Banks*, arXiv:2405.15179 (NeurIPS 2024). The formulas above follow the source.

## Key ideas and pitfalls

**Softmax normalizes over the `topk` selected scores only.** This is the line most people "accidentally write wrong": `F.softmax(top_k_logits)` receives the short post-topk slice, not the whole length-`V` logit vector. With a full softmax the rejected rows still receive weight, so each segment of `A` stops being a convex combination of the chosen rows — `check_selected_only_softmax` plants a rejected row pointing at `[100, 100]` so the mistake pushes a component to ~7.6. The flip side is an identity: the weights always sum to 1, so an all-ones bank forces `A` and `B` to all ones (last part of `check_bank_is_shared`).

**Which tied entry wins is unspecified in the source, so the problem has to pin it.** `torch.topk` makes no promise about ties (on this CPU, when a tie group straddles the cut it takes the *larger* index — e.g. `[0.5, 0.5, 9.0, -5.0]` with `topk=2` yields `[2, 1]`). This problem fixes the rule to **descending value, ascending index on ties** (a stable selection) and both the judge and the reference use it, so nothing drifts across backends. The notes keep the discrepancy on record: it is a reproducibility promise of the problem, not a peft semantic guarantee.

**The B side must be transposed before flattening.** The selection result is `(out_tile, r, vl)` and the target is `(d_out, r)`, so rows run as `o*vl + s` and the column is `j`; a plain `reshape(-1, r)` produces a tensor whose size happens to match but which has folded `r` into the row index — `check_b_transpose_layout` compares against exactly that wrong flattening. The A side must *not* be transposed: flattening `(r, in_tile, vl)` gives the intended tile-major order.

**`delta = B @ A`, not `A @ B`.** Shapes are `(d_out, r) @ (r, d_in)`; the mistake only hides on square layers with a matching rank. Expanding the forward `F.linear(F.linear(x, A), B)` gives `x @ Aᵀ @ Bᵀ = x @ (B@A)ᵀ`, so both routes must hold at once (`check_merge_equivalence`).

**There is no `alpha / r` scaling.** LoRA's `scaling = lora_alpha / r` does not exist in the VB-LoRA source; `get_delta_weight` is just `B @ A`, and `check_delta_no_extra_scaling` blocks that off with a `delta * r` counter-example.

**One bank feeds both sides.** `num_vectors` and `vector_length` are model-level hyperparameters, the last dimension of both logit tensors must equal the bank's row count, and `check_bank_is_shared` perturbs a single bank row to show that `A` and `B` can both move while `delta` still equals `B @ A`.

**Both degenerate ends deserve their own run.** At `topk = V` selection becomes a plain softmax-weighted sum over all rows (`check_topk_full` aligns it with an explicit `torch.softmax(logits)`) — the "soft" end used during training. At `topk = 1` the weight is identically 1 and the result is a hard `argmax` row (`check_topk_one`); gradient then reaches only the selected bank row and the logits themselves get none, which is exactly why the paper keeps a softmax over the top-k instead of a pure hard pick.

**Divisibility and consistency.** `d_in % vl` and `d_out % vl` must be 0, the `r` and `V` of both logit tensors must agree with the bank, and `0 < topk <= V` (`topk > V` is a `RuntimeError` inside torch, so the problem normalizes it to `ValueError`).

## How the judge cross-checks

- `_oracle` reimplements selection and assembly in plain numpy: `np.argsort(-logits, kind="stable")` for the first `topk`, `np.take_along_axis` for the scores, a hand-written max-shifted softmax, `bank[idx]` advanced indexing for the rows and a final sum, then `reshape(r, -1)` on the A side and `transpose(0, 2, 1).reshape(-1, r)` on the B side; it shares no operator with the reference.
- `check_against_oracle` runs six configurations (the `1×1×1` minimum, `topk=1`, `topk=V`, non-square layers, with bias, and pure float32 inputs) and compares all four return values `A`, `B`, `delta`, `y`; the remaining checks each guard one pitfall above.
- While authoring, real peft was used as a second oracle: a `tuners.vblora.layer.Linear(nn.Linear(in, out), {'def': bank}, ...)` with `vblora_logits_A/B` overwritten was compared against `solution.solve` over five `(in, out, r, V, vl, topk)` configurations — max deviation `6e-8` for `A`/`B`/`delta` and `2.4e-7` for the forward (from peft's fp32 path). No judged file imports peft.

## Complexity

Selection runs a top-k plus weighted sum over `r × (d_in/vl)` and `(d_out/vl) × r` logit vectors of length `V`, i.e. `O((r·d_in + r·d_out)/vl · V)` time (`O(V log topk)` with a heap, `O(V log V)` for a full stable sort); assembling `delta` costs `O(d_out · d_in)`; the unmerged forward only needs `O(N · r · (d_in + d_out))`, cheaper than materializing `delta` first. Trainable parameters are the two logit tensors, `r·(d_in + d_out)·V/vl`, plus the shared bank `V·vl` — independent of `d_out × d_in`, which is where the "extreme compression" comes from.
