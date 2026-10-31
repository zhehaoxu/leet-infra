# Reference: the modulation ⊙ and multiplicative merging

## Implementation

1. Validation order: `fan_in_fan_out` is bool → weight/A/B 2-D and finite → r and shapes (A `[r,in]`, B `[out,r]`, with out/in inferred from B's rows and A's columns) → x at least 1-D with last dim = in → fio direction matches the weight shape. Reject 0-D x (`torch.tensor(1.0)`) too.
2. **Separate the mathematical weight from the storage layout**: `w_math = weight` (fio=False) or `weight.T` (fio=True); all math happens in `[out, in]` and results are transposed back at the end. Upstream `forward` writes `transpose(weight, fio) * _prod_AB.T` with `_prod_AB = A.T @ B.T = (B@A).T` — expanding gives exactly `w_math ⊙ (B@A)`.
3. **Merging semantics**: upstream `_merge` computes `weight.data + base_weight * new_delta`, where `base_weight` is the pre-merge snapshot (the current W), so merged = `W + W⊙Δ`. Writing `W + delta` (additive) or `eff` (losing the base) are both wrong — both mines are in the mutation suite.
4. `hira_result = x @ eff.T`. Using the base weight (`x @ W.T`) is another mine: the HiRA residual goes through the **effective** weight, not the base forward.

## Verification

6 `check_*` functions cover: the hand-computed example (2×3, r=2, element-wise values of all four keys), 4 fixed-seed random parities (torch oracle `x @ (W*(B@A)).T`, alternating fio False/True with batched x), the fio=True layout roundtrip (same mathematical weight in both storages → equal hira_result, the three weight keys mutual transposes), 1-D x, the W=0 multiplicative-vs-additive anchor, the merge anchor (W=2I → merged=4I), and 10 invalid inputs.

The oracle recomputes directly in torch (`B@A`, `W*delta`, `x @ eff.T` written out independently in the tests) — same math as the solution but a different code path; the element-wise hand example pins the semantics.

Mutation testing (5/6 caught + 1 equivalent): `⊙`→`+` (multiplicative → additive; example and anchors all catch), merged as `W+Δ`, merged losing the base, lost fio transpose, residual via the base weight. Removing `if r < 1` survives — r is inferred from `A.shape[0]`, so the shape validation already rejects empty matrices; the mutation is observationally equivalent (documented in notes).

## Scope and sources

Upstream huggingface/peft `src/peft/tuners/hira/layer.py`:

- `HiraLayer._merge` (L85-104): `new_delta += get_delta_weight(...)`; `merged_weight = weight.data + base_weight * new_delta` — the origin of multiplicative merging; `_get_base_weight_before_merge` snapshot semantics (W in this problem is the snapshot).
- `Linear.forward` (L245-273): `_prod_AB = torch.mm(hira_A.T, hira_B.T)`; `F.linear(dropout_sub, transpose(weight, fio) * _prod_AB.T)` — effective weight = mathematical weight ⊙ (B@A).
- `Linear.get_delta_weight` (L213-243): `transpose((weight_B @ weight_A), fan_in_fan_out)` — the delta_weight layout convention.
- `reset_hira_parameters` (L149-165): Kaiming/Gaussian A and zeroed B init (B=0 ⇒ identity after merge); training-side details out of scope.

Deliberate deviations:

- **No model, no dropout**: upstream forward applies `hira_dropout(x)` (Identity at inference); here x goes straight in. RoPE/attention wiring, Embedding/Conv variants out of scope.
- Explicit return contract: all four keys returned (upstream only holds delta/effective implicitly); layout pinned as "same storage layout as the input weight".
- `fan_in_fan_out` covers only the Linear-side Conv1D layout (`[in, out]` storage and transpose); `safe_merge` NaN checks and multi-adapter accumulation (`new_delta += ...`) out of scope.
