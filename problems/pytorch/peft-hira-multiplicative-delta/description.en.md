# HiRA Multiplicative High-Rank Adaptation: W⊙(BA) and Merging

## Problem

LoRA-family adapters update the weight **additively**: W' = W + BA. HiRA (huggingface/peft `tuners/hira`) couples the low-rank product differently — as a **modulation matrix** applied element-wise:

$$W' = W \odot (1 + BA) = W + W \odot (BA)$$

At forward time the effective weight is $W_{\text{eff}} = W \odot (BA)$ and the HiRA residual is $x W_{\text{eff}}^T$ — **not** the base output plus $x(BA)^T$. Merging in upstream code reads `merged_weight = weight + base_weight_snapshot * delta`; since the snapshot is the current base weight, the semantics are $W(1+BA)$. This one-word difference is what lets HiRA preserve the full rank of the base weight (the origin of "high-rank adaptation"), and it means: wherever W has a zero, no value of BA can ever move that entry.

Given the weight, two low-rank factors, and an input, return four quantities — Δ, the effective weight, the HiRA residual, and the merged weight — while handling the `fan_in_fan_out` layout (HuggingFace Conv1D stores weights as `[in, out]`).

## Task

Implement `solve(weight, hira_A, hira_B, x, fan_in_fan_out=False)`:

- `weight`: 2-D float tensor. Shape `[out, in]` when `fan_in_fan_out=False`; `[in, out]` when `True` (Conv1D storage; the mathematical weight is its transpose);
- `hira_A`: `[r, in]` with `r >= 1`; `hira_B`: `[out, r]`;
- `x`: tensor with at least one dimension whose last dim equals `in`; batch `[batch, in]` or single token `[in]`;
- `fan_in_fan_out`: bool.

Return a `dict` with keys `delta_weight` (`B@A`), `effective_weight` (`W⊙(B@A)`), `hira_result` (`x @ effective^T`, last dim `out`), `merged_weight` (`W + W⊙(B@A)`). **All three weight keys follow the input storage layout**: `[in, out]` when `fan_in_fan_out=True` (the transpose of the mathematical layout); `hira_result` is unaffected.

## Exact rules

1. Recover the mathematical weight: `W = weight` (fio=False) or `weight.T` (fio=True), always `[out, in]`.
2. `delta = B @ A`, `[out, in]`.
3. `effective = W ⊙ delta` (element-wise — not a matmul, not an addition).
4. `hira_result = x @ effective.T`, shape `[..., out]`.
5. `merged = W + effective` (equivalently `W⊙(1+delta)`).
6. With fio=True, transpose `delta/effective/merged` back to `[in, out]` before returning.

## Example

```text
W = [[1, 2, 0.5], [0, -1, 2]] (out=2, in=3)
A = [[1, 0, -1], [0, 1, 1]] (r=2)
B = [[1, 2], [0, -1]]
x = [[1, 1, 1]]

B@A = [[1, 2, 1], [0, -1, -1]]
W⊙(B@A) = [[1, 4, 0.5], [0, 1, -2]]
hira_result = x @ eff.T = [[5.5, -1.0]]
merged = W + W⊙(B@A) = [[2, 6, 1], [0, 0, 0]]
```

Two semantic anchors:

- Set `W` to all zeros with Δ = I and x = I: `effective/merged/hira_result` are all zero — under the **multiplicative** semantics W=0 freezes everything; additive LoRA would return BA=I.
- `W = [[2,0],[0,2]]` with Δ = [[1,1],[1,1]]: merged = `[[4,0],[0,4]]` — off-diagonal entries stay 0 because W is 0 there; additive merging would give `[[3,1],[1,3]]`.

## Constraints

- All inputs finite; do not modify inputs; invalid inputs (shape mismatches, non-bool fio, fio/weight shape contradiction, x last-dim mismatch, 0-dim x, non-finite values) raise `ValueError`.
