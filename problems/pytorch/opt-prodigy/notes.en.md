# Prodigy Adaptive Learning Rate (D-Adaptation) Explained

## Source
Defazio & Mishchenko, *Learning-Rate-Free Learning by D-Adaptation* (arXiv:2301.07733), and its practical Adam implementation Prodigy. The reference implementation follows `Prodigy` from kozistr/pytorch-optimizer (defaults `lr=1.0`, `betas=(0.9,0.999)`, `beta3=√beta2`, `d0=1e-6`, `d_coef=1.0`, `growth_rate=inf`, `bias_correction=False`, `safeguard_warmup=False`, `eps=1e-8`, `weight_decouple=True`). On CPU `foreach` is not triggered and the per-parameter path is taken; this explanation describes that path.

## Core idea: online estimation of "the distance D to the optimum"
Adam's step size \(lr\) needs manual tuning. D-Adaptation observes that if the distance \(D\) from the initial parameters to the optimal solution were known, the optimal step size could be derived from \(D\). Prodigy uses a "dual-style" estimate to approximate \(D\) online:

- \(\mathit{num}\) (a group-level scalar, EMA across steps) accumulates each step's \( \langle g,\ p_0-p\rangle \) — the inner product of the gradient with the "displacement relative to the initial point", proportional to the decrease in function value;
- \(\mathit{den}\) (recomputed every step) is \(\sum|s|\), where \(s\) is the \(\beta_3\)-EMA of the gradient, proportional to the "cost" of the step;
- distance estimate \(\hat d = d_{coef}\cdot \mathit{num}/\mathit{den}\); then multiply \(d\) into the effective step size \(d_{lr}=d\cdot lr\).

As a result \(lr\) can stay at 1, and \(d\) grows to the appropriate scale by itself.

## Why it is hard to get right (pitfalls)
1. **Two passes**: first compute statistics for **all** parameters (updating \(m,v,s,\mathit{num},\mathit{den}\)), **then** compute \(d\) once, **and only then** does the second pass actually update \(p\). A single-pass implementation is wrong.
2. **Old \(d_{lr}\) vs new \(d\)**: the second pass's step size `value=-d_lr` uses the \(d_{lr}\) computed from the **old \(d\) at the start of the step**, while the denominator `√v + d·eps` uses the **new \(d\)** just updated.
3. **`num` decays before accumulating**: at the start of each step \(\mathit{num}\leftarrow\beta_3\mathit{num}\), and only then `+=`; whereas \(\mathit{den}\) is zeroed and recomputed every step.
4. **\(d\) is monotone, with a growth_rate cap**: only on the first step, when \(d==d_0\), is it `max(d0, d̂)`; afterwards \(d_{\max}=\max(d_{\max},\hat d)\) and \(d=\min(d_{\max}, d\cdot\)growth_rate\()\). With `growth_rate=inf`, \(d=d_{\max}\) (monotonically non-decreasing).
5. **The moments also carry \(d\)**: \(m\leftarrow\beta_1 m + d(1-\beta_1)g\) and \(v\leftarrow\beta_2 v + d^2(1-\beta_2)g^2\) (note the \(d\), \(d^2\) coefficients, unlike plain Adam).

## Grading details
The grader uses an independent recomputation `_oracle_prodigy` (the two passes written in list form, with explicit `num/den` scalars), compared with the reference solution under multiple shapes and multiple steps of random gradients with `allclose(atol=1e-5, rtol=1e-5)`; in this collection the two were measured to be **exactly identical** element-wise (maxdiff=0), because both take the same CPU scalar path. `check_none_grad_param_skipped` verifies that parameters without gradients are skipped in both passes and that not even \(p_0/s\) is created.

## Complexity
Two passes of \(O(\text{numel})\) per step; extra state \(m,v,s,p_0\) (about four copies, one more \(s\) and the snapshot \(p_0\) than Adam).
