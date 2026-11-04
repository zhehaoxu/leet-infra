## Problem

SA-Solver (Xue et al. 2023, arXiv 2309.05019) casts diffusion sampling as a pair of stochastic multistep methods: the **SA-Predictor** (stochastic Adams-Bashforth, explicit) takes the forward step, and the **SA-Corrector** (stochastic Adams-Moulton, implicit) uses the freshly obtained network output at the next call to **retroactively correct** the previous landing point. `SASolverScheduler.step()` runs five beats per call in a fixed order — "convert → corrector → history shift → warmup order selection → predictor" — and threads N calls into one sampling trajectory through a set of internal states (`model_outputs` history, `last_sample`, `last_noise`, `lower_order_nums`, `this_corrector_order`). This problem asks you to implement **one call** of that state machine, with the state passed explicitly via `state`.

Judge inputs:

- `model_output` / `sample`: floating-point tensors of the same shape;
- `sigmas`: 1-D float tensor, the table `[sigma_0, …, sigma_{N−1}, sigma_last]` produced by the official `set_timesteps` (length N+1, N ≥ 2), **strictly positive** (no upstream schedule ends in 0), finite, non-increasing;
- `step_index`: the current index, `0 ≤ step_index ≤ N−1`;
- `state`: `None` = cold start (including a mid-chain cold start as in image-to-image); otherwise the `state` dict returned by the previous call;
- `noise`: **required on every call** (officially drawn from a generator each step; an explicit input here), shape must match `model_output` — even when tau = 0 (the official code still draws it and stashes it for the next call's corrector);
- `tau` / `tau_corrector`: stochasticity coefficients (the official `tau_func(t)` outputs only 0 or 1). The corrector evaluates it at the **previous** timestep and the predictor at the **current** timestep, hence two separate inputs; `tau_corrector` is ignored on calls without a corrector;
- `predictor_order` / `corrector_order` ∈ [1, 4]: configured orders; `lower_order_final`: the end-of-schedule order-dropping switch (officially True by default);
- `prediction_type` ∈ {`epsilon`, `v_prediction`}.

This problem pins the official default path: `algorithm_type = data_prediction` (predict_x0 = True), no thresholding, no flow branch.

## Rules

**Sigma conventions** (recomputed from `sigmas` on every call, pure function): alpha = 1/sqrt(sigma²+1), sigma_t = sigma·alpha, lambda = log alpha − log sigma_t. Mathematically lambda ≡ −log sigma and e^lambda ≡ 1/sigma (an identity of the VE convention; the official code still walks the two-log path, which this problem keeps verbatim).

**convert** (`prediction_type`): at sigma_now = `sigmas[step_index]`, turn the network output into an x0 prediction:

- `epsilon`: x0 = (sample − sigma_t·mo)/alpha = sqrt(sigma²+1)·sample − sigma·mo;
- `v_prediction`: x0 = alpha·sample − sigma_t·mo.

**corrector** (only when `step_index > 0` and `state` is given): the order tc is the `state["this_corrector_order"]` **saved by the previous call**, and tau is `tau_corrector`:

- sigma_t at `sigmas[step_index]`, sigma_s0 at `sigmas[step_index−1]`; h = lambda_t − lambda_s0;
- lambda node list `[lambda(sigmas[step_index−k]), k = 0..tc−1]`;
- coefficients = Lagrange(tc−1, nodes) × positive-exponential integrals ∫exp(x(1+tau²))·x^(tc−1−j)dx over [lambda_s0, lambda_t] (verbatim from the official `get_coefficients_exponential_positive` substitution);
- **tc = 2 correction** (the UniPC-style O(h³) term): delta = exp((1+tau²)lambda_t)·(h/2 − (h(1+tau²) − 1 + exp(−(1+tau²)h))/((1+tau²)²·h)), coefs[0] += delta, coefs[1] −= delta;
- model-output list = `state["model_outputs"]` (**before** the shift) + [x0], taking the last tc entries;
- gradient = Σ_k (1+tau²)·sigma_t·exp(−tau²·lambda_t)·coefs[k]·mo_list[−(k+1)];
- noise_part = sigma_t·sqrt(1 − exp(−2tau²h))·`state["last_noise"]` — **the noise injected by the previous predictor**, keeping the randomness of the two sides consistent;
- corrected = exp(−tau²h)·(sigma_t/sigma_s0)·`state["last_sample"]` + gradient + noise_part (the launch point is the sample from **before** the previous predictor).

**history shift**: `model_outputs = old[1:] + [x0]`, length M = max(predictor_order, corrector_order − 1) unchanged (the −1 exactly makes room for the corrector's x0).

**warmup order selection**: with `lower_order_final`, tp = min(predictor_order, N − step_index) and tc_new = min(corrector_order, N − step_index + 1); otherwise both take the configured orders. Then uniformly tp = min(tp, lower_order_nums + 1) and tc_new = min(tc_new, lower_order_nums + 2). **On the final step (step_index = N−1) the predictor always drops to order 1**; the warmup guarantees the None head of the history is never read.

**predictor**: order tp, tau from `tau`:

- sigma_t at `sigmas[step_index+1]`, sigma_s0 at `sigmas[step_index]`; the node list is analogous;
- **tp = 2 correction**: delta = exp((1+tau²)lambda_t)·(h²/2 − (h(1+tau²) − 1 + exp(−(1+tau²)h))/(1+tau²)²)/(lambda_s0 − lambda(sigmas[step_index−1])) — the denominator is the **adjacent lambda gap**, a different formula from the corrector's; step 0 never hits it because warmup forces tp = 1;
- model-output list = the **post-shift** history (x0 at the tail), taking the last tp entries;
- gradient / noise_part have the same formulas as the corrector, with `noise` from **this call**;
- prev = exp(−tau²h)·(sigma_t/sigma_s0)·(corrected or sample) + gradient + noise_part.

**Return dict**: `prev_sample` / `pred_original_sample` (float32), `corrected_sample` (float32, None when no corrector ran), `used_corrector` (bool), `this_predictor_order` / `this_corrector_order` (the actual post-warmup orders of this call), `predictor_coefficients` / `corrector_coefficients` (lists of floats; the latter None when no corrector ran), and `state`. The new `state` = {`model_outputs`, `last_sample` (corrected or sample), `last_noise` (this call's noise), `lower_order_nums` (+1 while below M), `this_corrector_order` = tc_new}. Validation failures always raise `ValueError`.

## Example

```
sigmas = [4/3, 3/4, 1/4], orders 2/2, lower_order_final = True, tau = 0

step 0 (state = None): mo = 0.3, sample = 1.0, noise = 0.9
  alpha(4/3) = 3/5, sigma_t(4/3) = 4/5
  x0 = sqrt(25/9)*1.0 - (4/3)*0.3 = 5/3 - 2/5 = 19/15
  tp = min(2, 2-0) capped by (0+1) -> 1 (warmup), tc_new = min(2, 3, 0+2) = 2
  coef[0] = exp(lam(3/4)) - exp(lam(4/3)) = 1/(3/4) - 1/(4/3) = 4/3 - 3/4 = 7/12
  noise_part = (3/5)*sqrt(1-exp(0))*noise = 0   (tau = 0, but noise still stored)
  prev = (3/5)/(4/5)*1.0 + (3/5)*(7/12)*(19/15) = 3/4 + 133/300 = 179/150 = 1.1933
  state = {model_outputs: [None, 19/15], last_sample: 1.0, last_noise: 0.9,
           lower_order_nums: 1, this_corrector_order: 2}

step 1 (state from step 0): mo = -0.2, sample = 179/150, noise = 0.9, tau_corrector = 0
  x0 = (5/4)*(179/150) - (3/4)*(-0.2) = 197/120 = 1.6417
  corrector (tc = 2 saved): coefs = [0.3836, 0.1998]  (order-2 mod, delta = 0.0641)
  corrected = (3/5)/(4/5)*1.0 + (3/5)*(0.3836*(197/120) + 0.1998*(19/15)) = 1.2796
  predictor: lower_order_final -> tp = min(2, 2-1) = 1 (final step drops to order 1)
  coef[0] = 1/(1/4) - 1/(3/4) = 4 - 4/3 = 8/3
  prev = (1/sqrt(17))/(3/5)*1.2796 + (1/sqrt(17))*(8/3)*(197/120) = 1.5790
```
