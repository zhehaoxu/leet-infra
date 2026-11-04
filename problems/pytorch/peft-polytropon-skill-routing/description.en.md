# Polytropon skill-inventory routing

Polytropon ("Combining Modular Skills in Multitask Learning") and Multi-Head Adapter
Routing (MHR, "Multi-Head Adapter Routing for Cross-Task Generalization") ship in
HF PEFT (`peft/tuners/poly`). Each layer keeps an *inventory* of `n_skills` LoRAs;
a per-task routing function learns the mixture. With `n_splits > 1` each LoRA's rank
is split into disjoint blocks and every block routes independently (MHR).

## Parameters and shapes

- `poly_lora_A`: (n_splits, n_skills, in//n_splits, r)
- `poly_lora_B`: (n_splits, n_skills, r, out//n_splits)
- router `module_logits`: (n_tasks, n_splits·n_skills), `uniform_(-1e-3, 1e-3)`

## Initialization (RNG domain)

A/B are initialized per (skill, split) block with **skill-outer, split-inner loops**:

- A: `kaiming_uniform_` on a `(r, d_in)` tensor (fan_in = d_in = in//n_splits,
  a=√5 → bound = 1/√d_in), **transposed** into `A[split, skill]`;
- B: zeros when `init_weights=True` (zero delta at training start); otherwise
  `kaiming_uniform_` on `(d_out, r)` (fan_in = r), transposed into place;
- router: `uniform_(-1e-3, 1e-3)` after all A blocks consumed the generator.

Floating-point subtlety: with `a = math.sqrt(5)`, `1 + a·a` is not exactly 6
(error ~1e-16), so the bound carries the same error — RNG-stream replays must use
the identical formula or the uniform draws differ at the bit level.

## Routing (inference branch)

```
logits = module_logits[task_ids].view(-1, n_splits, n_skills)
s = sigmoid(logits)
weights = s / (s.sum(-1, keepdim=True) + 1e-12)
```

Three judge-sensitive points:

- **Normalized sigmoids, not softmax** — numerically distinct;
- **The +1e-12 EPS guard**: when sigmoids underflow to 0 it prevents 0/0 = NaN
  (judgeable: an all-`-800` logits case must give finite, zero weights);
- An out-of-range task id raises `ValueError` (note the `>=` boundary).

The training-branch `RelaxedBernoulli.rsample` is stochastic and out of scope.

## Forward combination

```
A = einsum("bqs,qsdr->bqdr", w, poly_lora_A).reshape(bs, in, r)
B = einsum("bqs,qsrd->bqrd", w, poly_lora_B).transpose(1, 2).reshape(bs, r, out)
delta = x.bmm(A).bmm(B) / r        # the scale is exactly 1/r, not α/r or 1/√r
```

Reassembly semantics: A's reshape places split 0's input block in the first
`in//n_splits` columns; B transposes (1,2) then reshapes, so output columns are
split-major. Identity: `A_mixed[:, :in//ns]` depends only on split 0's weighted sum.

## Parameter budget

lora = ns·sk·(in//ns)·r + ns·sk·r·(out//ns); router = n_tasks·ns·sk;
`n_splits=1` reduces to a plain multi-LoRA inventory.

## Judging domain

Single entry `solve(mode, **kw)`, five modes: `init_params` / `route` / `combine` /
`forward_delta` / `param_count`. Explicit generator RNG domain; float64 anchors.
