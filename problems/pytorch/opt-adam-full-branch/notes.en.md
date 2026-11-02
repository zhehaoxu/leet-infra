# notes (opt-adam-full-branch)

## Source

- Code: torch 2.14.1 in the local judging environment, `torch/optim/adam.py` (same as
  upstream main).
  - `Adam.__init__` (34-120): `not 0.0 <= lr` (58), eps (60), betas[0] (62), betas[1] (64),
    weight_decay (66), plus the "both floats or both Tensors" rule at 68-71.
  - `_init_group` (139-208): the state is **lazily created** (only when `len(state) == 0`) —
    `step` is `torch.tensor(0.0)` (168), `exp_avg` / `exp_avg_sq` start at zeros, and
    `max_exp_avg_sq` starts at zeros too (187, only under amsgrad).
  - `_single_tensor_adam` (347-545), scalar path: 397 maximize negation, 414 `step_t += 1`,
    426/428 weight_decay, 456 `exp_avg.lerp_`, 471/475
    `exp_avg_sq.mul_().addcmul_()`, 530-533 bias correction and `step_size`, 535
    `bias_correction2_sqrt`, 539 the amsgrad `torch.maximum(..., out=)`, 542/544 denom, 546
    `param.addcdiv_`.
  - `_multi_tensor_adam` (553-) was checked for semantics only: same negation first, same
    lerp/mul-add, same `bias_correction2_sqrt` and eps placement.
- Cross-check: `check_matches_torch_optim` uses `torch.optim.Adam` itself as the oracle
  (8 configs × 4 trials × 8 steps, two trials forcing `grad=None`, one all-None), matching at
  1e-12 in float64; the measured worst absolute deviation is 8.9e-16.
- Another 1200 random configurations (lr/betas/eps/wd/amsgrad/maximize/state all random) were
  used as evidence for the equivalence mutations below.

## Traps in the upstream implementation

1. **`step` increments before use** (line 414): the state starts at 0, so the first update
   uses `t = 1` and the k-th update uses `t = k`. Computing first and incrementing afterwards
   throws the bias correction off by a whole step.
2. **A parameter with no gradient never gets state** (`_init_group` only enters the body when
   `p.grad is not None`), so **a skipped step does not advance `step`**. Hence the first real
   gradient of `[None, g]` gets the `t = 1` bias correction rather than `t = 2` — the easiest
   thing to miss in this problem.
3. **All five gates use `not 0.0 <= x`**: `0.0 <= nan` is false, so **NaN is rejected** too
   and the message prints `nan`. `betas` additionally requires `< 1.0`, so `beta = 1.0` is
   illegal (many re-implementations only check `>= 0`).
4. **`eps` is added after `sqrt(v)/sqrt(bc2)`** (the `.add_(eps)` on line 544); it is not
   scaled by `sqrt(bc2)`. So the first displacement is `-g/(|g|+1e-8)` rather than `-g/|g|`:
   with `eps=1e-8` it is `0.099999998`, a 2e-9 difference. The "eps is also bias-corrected"
   variant of the paper formula differs by 6.3e-10 — three times smaller — so a hard-coded
   expectation at `atol=1e-12` pins the placement exactly.
5. **amsgrad updates its maximum before using it** (539 comes before 542), and that state is
   initialised to **zeros** (not a copy of `v`). Since `v` is always non-negative, the first
   step has `max(0, v) = v`, so "init to zero" and "init to v" are indistinguishable there —
   only the zero initialisation keeps the old maximum when `v` later shrinks (the problem pins
   this with a "large gradient then small gradient" pair).
6. **maximize negates before weight_decay** (397 vs 426): the effective gradient is `-g + λp`,
   not `-(g + λp)`. With `p=[1,-2]`, `g=[0.5,-1]`, `λ=0.5` the effective gradient is exactly
   zero, so a correct implementation leaves the parameters alone while a reordered one moves
   them.
7. **Adam's `weight_decay` is L2-style**: λp joins the gradient and then enters **both**
   moments, so the decay is divided by `1/√v` and differs per parameter; `opt-adamw` (182) is
   the decoupled form `p *= (1 - lr·λ)`, independent of the adaptive denominator. torch 2.x
   also exposes a `decoupled_weight_decay=True` switch (present in 2.14.1, going through
   `param.mul_(1 - lr*weight_decay)`); this problem covers the default L2 path only.
8. **The two bias corrections are folded differently**: `step_size = lr/bc1` is folded first
   and `1/sqrt(bc2)` is folded into `denom`, ending with
   `addcdiv_(m, denom, value=-step_size)`. That is algebraically
   `lr·m̂/(√v̂+eps)`, but `eps` is not scaled by `sqrt(bc2)` (see item 4).
9. **Lazy state is all-or-nothing**: once created, `step/exp_avg/exp_avg_sq` (plus
   `max_exp_avg_sq` under amsgrad) are all present; there is no half-built state.
10. **`lr = 0` is legal**: `step_size = 0`, so the parameters stay put while `m`, `v` and
    `step` keep advancing.
11. **`betas` may be zero**: `β₁=0` gives `m = g`, `β₂=0` gives `v = g²`, and both bias
    corrections equal 1.
12. **`eps = 0` is legal too**: the normalisation is pure `sqrt(v̂)`; if a step's gradient is
    exactly zero then `v = 0`, `denom = 0` and `m/denom` is `0/0 = nan` — upstream has no
    guard at all. That is why the "`maximize + wd` makes the effective gradient exactly zero"
    case keeps the default `eps`, so its displacement is 0 rather than nan.
13. **A `betas` sequence that is too short raises IndexError upstream** (`betas[1]` is out of
    range), not ValueError.

## Normalizations in this problem

- **`betas` takes only two floats**: upstream also allows a Tensor pair and rejects a mixed
  pair with `"betas must be either both floats or both Tensors"`; this problem fixes them to
  floats and raises `ValueError` when the length is not 2 (upstream raises `IndexError`).
- **`state` is made explicit**: upstream keeps a dict with a scalar-tensor `step`; here it is
  an input dict whose `step` is normalised to an int (the number of updates already done).
  With `amsgrad=False` the key `max_exp_avg_sq` is still required in the input but is **not
  read**, and the return value reports zeros (upstream's state has no such key at all).
- **`grads[t] = None` maps to upstream's "filter by parameter"**: from a single-parameter
  point of view it is the same as "`p.grad is None` on this step".
- **Two new observation surfaces, `steps` and `deltas`**: upstream exposes neither "which t
  this step used" nor "what the actual displacement was". Recording both makes the
  "skipped steps do not advance the counter" and "where eps sits" rules directly assertable
  instead of inferable from float drift.
- **Real tensors, one group, Python scalar lr/eps/wd only**: the `view_as_real` complex path,
  per-group hyper-parameters and Tensor lr (with its `capturable`/`foreach` constraints) are
  out of scope.

## Mutations

33 mutations: **all 29 substantive ones are CAUGHT**, 4 are classified as equivalent with
evidence (1200 random configurations: lr ∈ {0, 0.01, 0.1, 0.3}, 4 beta pairs,
eps ∈ {0, 1e-3, 1e-8}, wd ∈ {0, 0.01, 0.2}, random amsgrad/maximize, 30% with a random
state — 0 differences):

- `m-no-param-clone` (`p = params.clone()` → `p = params`): `p` is only **rebound** by
  `p = p + delta`, never mutated in place, and `path` clones each snapshot.
- `m-delta-no-clone` (`deltas.append(delta.clone())` → `deltas.append(delta)`): `delta` is a
  fresh tensor each step and `p` is formed by **adding** it into a new tensor, so it shares
  storage with no returned field.
- `m-momentum-lerp` (`β₁m+(1-β₁)g` → `m + (1-β₁)(g-m)`): a mathematical identity with a
  different rounding path; the measured worst deviation is **1.06e-15**, far inside the
  1e-12 contract.
- `m-var-order` (`(1-β₂)·g·g` → `g·g·(1-β₂)`): different multiplication association, worst
  deviation **4.4e-16**.

No substantive mutation survived the first round; the thing that did break was a **wrong test
assertion**: `check_skip_none_does_not_advance_step` treated `path[3]` as "still the original
parameters", but `path[3]` is the snapshot of the second skipped step (equal to `path[2]`, not
`path[0]`), so the baseline failed first. After fixing it to `path[3] == path[2]` the suite is
16/16. Another reminder that "the suite must be green before mutations mean anything".

## Overlap check

- Against `adam-step` (75): that one is a single step with `amsgrad=False` and no
  weight_decay; this problem adds multiple steps, the amsgrad running maximum, L2 decay,
  maximize, the five constructor gates and state resumption.
- Against `opt-adamw` (182): AdamW uses decoupled decay (`p -= lr·λ·p`, unrelated to `1/√v`)
  while this problem uses L2 (λp enters both moments and gets divided by `1/√v`). The two form
  a deliberate contrast pair, and the notes spell out why L2 is not a uniform shrinkage inside
  Adam.
- Against `opt-adamp` (189), `opt-radam` (187), `opt-adamuon` (267) and
  `opt-adabelief`: those are variant formulas (projection / rectification / muon-isation);
  this problem is vanilla Adam's full branch matrix and ledger.
- Against `opt-sgd-full-branch` (738): the SGD member of the same "full branch" series, with
  no adaptive denominator and no moments. Both share the "a None step creates no state"
  rule, but the consequence differs (SGD delays the first buffer, Adam does not advance the
  bias-correction `t`).
- Against `opt-clip-accum` (197): that one is gradient accumulation plus clipping, with no
  moment estimates.
