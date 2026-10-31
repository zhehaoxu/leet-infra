# Reference Approach: Where the Clamp Goes, Bias Isolation, and Ties Caused by Saturation

## Implementation

1. `clamped = torch.tanh(logits / tau) * tau` for `tau > 0`; pass `logits` straight through for `tau == 0` instead of dividing by zero.
2. `scores = torch.sigmoid(clamped)`.
3. `selection = scores if bias is None else scores + bias`.
4. `indices = torch.argsort(selection, dim=-1, descending=True, stable=True)[:, :k]`. The "lower expert index wins ties" convention is carried by `stable=True`; on the pure Python side the same rule is `sorted(range(E), key=lambda i: (-sel[i], i))`.
5. `picked = scores.gather(-1, indices)` — note it gathers from **scores**, not from selection.
6. `total = picked.sum(-1, keepdim=True)`; `weights = torch.where(total > 0, picked / total.clamp_min(1e-300), full_like(picked, 1.0 / k))`. The `clamp_min` here only keeps the untaken branch of `where` from actually computing 0/0; the taken branch is numerically untouched.

Common mistakes:

- **Clamping after the sigmoid.** `tau*tanh(z/tau)` caps the logits; applying it to probabilities squashes an already-squeezed $s\in(0,1)$ and changes what `tau` means entirely. The `-800/-900` underflow case is precisely there to separate the two.
- **Forgetting the division by `tau`**, i.e. writing `tanh(z) * tau`: the cap magnitude looks right but the saturation point is wrong.
- **Normalizing `s + bias`.** The bias is a load-balancing correction term; letting it into the weights silently changes the actual coefficient on each expert output, which is exactly what auxiliary-loss-free routing warns against.
- **Missing the `tau = 0` branch**, so `logits / 0` becomes inf and multiplying by 0 yields NaN.
- **No underflow fallback.** With `tau = 0` and a logit like `-800`, the sigmoid is exactly 0, the selected sum is 0, and `picked / total` gives NaN instead of the contractual `1/k`.

## Correctness checks

The 7 `check_*` functions cover: the hand example (including "a bias swaps the pick while weights still normalize `s`"), 64 randomized differential cases over the full seed × tau × with/without-bias grid with `k` and logit scale varying by seed, saturation and ties (`tanh(20)==tanh(21)`, `tau=1e6` selecting like the disabled clamp, `tau=1e-3` letting bias decide the pick), bias isolation (lifting expert 1 into the selection, constant bias changing nothing), boundaries and underflow (`k=E`, all-zero ties, `[-800,-900,5]` giving score 0 with weights `[1,0]`, `[-800,-900]` giving uniform `1/k`, noncontiguous input), invalid inputs, and the maximum shape `8x8, k=8`.

The oracle runs entirely on pure Python scalars: `math.tanh`, a hand-written stable sigmoid (negatives go through `exp(z)/(1+exp(z))`, otherwise `math.exp(800)` overflows), selection by `sorted(key=lambda i: (-sel[i], i))`, and normalization with `math.fsum`. It borrows no torch `sigmoid`/`argsort`/`gather`.

Mutation testing: clamping after the sigmoid, dropping the division by `tau`, dropping the `tau=0` branch, mixing bias into the weights, gathering from selection, and removing the underflow fallback — all six mutants are caught, and "remove the underflow fallback" is caught only by `edges_and_underflow`, which shows that case is necessary rather than redundant. Two further mutants (an `argsort` without `stable=True`, and `torch.topk`) happen to agree with the contractual order in this machine's environment, so they are not distinguishable; that is not a gap in the tests — the problem constrains the outcome ("ties go to the lower index") and does not pin which API you use.

## Scope and sources

The mechanism corresponds directly to `models/laguna/modeling_laguna.py::LagunaTopKRouter.forward` in the transformers 5.x source available to this repository's judge:

```text
if self.router_logit_softcapping > 0.0:
    router_logits = torch.tanh(router_logits / self.router_logit_softcapping) * self.router_logit_softcapping
routing_scores = torch.sigmoid(router_logits)
scores_for_selection = routing_scores + self.e_score_correction_bias...
_, selected_experts = torch.topk(scores_for_selection, self.top_k, dim=-1)
routing_weights = routing_scores.gather(-1, selected_experts)
routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
```

This problem extracts that chain as explicit rules: the soft cap becomes the parameter `tau`, `tau = 0` plays the role of the config's "soft capping disabled", the bias only enters `scores_for_selection`, and the gather reads from `routing_scores`. The same `tanh(z/cap)*cap` shape also shows up in gemma2's `final_logit_softcapping` and in its attention `score_mod`, which is why it is treated as a general stability device rather than a router-specific trick.

Real implementations use `torch.topk`; here ties are pinned to "lower index first" because the judge needs reproducible expectations. The uniform `1/k` under underflow, `tau=0` disabling the clamp, and not updating the bias at all are conventions of this problem, not defaults of the upstream code. Grouped top-k lives in `moe-aux-free-routing`, and this problem covers no bias sign updates, no z-loss or load-balancing loss, and nothing beyond the forward pass.
