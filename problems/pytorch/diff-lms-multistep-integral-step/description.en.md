## Problem

LMS-Discrete (`LMSDiscreteScheduler`) is the classic Linear Multi-Step sampler: it accumulates **historical derivatives** of the ODE right-hand side `(sample − x0)/sigma`, builds a Lagrange interpolation polynomial through the most recent data points, and integrates it over sigma-space to extrapolate one step to the next sigma. Its entire mechanics condense into two things: (a) the coefficients are **integrals of Lagrange bases over sigma-space** (the official code computes them on the fly with `scipy.integrate.quad`, recomputed per coefficient on every call); (b) the **derivative history** with append-and-evict capping (keep the most recent `order` entries). This problem asks you to implement one call, with the history passed explicitly via `state`.

Judge inputs:

- `model_output` / `sample`: floating-point tensors of the same shape;
- `sigmas`: 1-D float tensor, the official `set_timesteps` table `[sigma_0, …, sigma_{N−1}, 0]` (length N+1, N ≥ 2), with the first N entries positive, the last exactly 0, finite and non-increasing (the same 0 tail as DPMSolver/KDPM2);
- `step_index`: the current index, `0 ≤ step_index ≤ N−1`;
- `state`: `None` = cold start; otherwise the `state` dict returned by the previous call (`step_index = 0` must pass None);
- `prediction_type` ∈ {`epsilon`, `v_prediction`, `sample`} (all three branches exist upstream);
- `order` ∈ [1, 4]: the linear multistep order (officially 4 by default).

## Rules

Four beats per call (verbatim from the official `step`, L545-623):

**1. convert**: at sigma = `sigmas[step_index]`, turn the network output into an x0 prediction:

- `epsilon`: x0 = sample − sigma·mo;
- `v_prediction`: x0 = mo·(−sigma/sqrt(sigma²+1)) + sample/(sigma²+1);
- `sample`: x0 = mo.

**2. derivative and history**: derivative = (sample − x0)/sigma (the denominator is the current sigma, finite because the first N sigmas are positive); append it to the history, popping from the head when over the cap — keep the most recent `order` entries.

**3. LMS coefficients**: `order_used = min(step_index + 1, order)` (warmup: automatically lower order while the history is short). For curr_order = c ∈ [0, order_used), take the official `get_lms_coefficient(order_used, step_index, c)`:

- the integrand ℓ_c(tau) = Π_{k∈[0,order_used), k≠c} (tau − sigma_{t−k}) / (sigma_{t−c} − sigma_{t−k}) — a Lagrange basis (1 at node k=c, 0 at the others);
- integrated over [`sigmas[step_index]`, `sigmas[step_index+1]`]. **The sigma table decreases → the interval is reversed → the integral is negative** (same semantics as the official `quad`); the final step integrates over [sigma_{N−1}, 0], ending at 0;
- this problem uses **power-coefficient expansion and analytic integration** (ℓ_c is a polynomial of degree ≤ 3; expand and integrate term by term, ∫tauⁱ = (b^{i+1} − a^{i+1})/(i+1)) — mathematically equivalent to the official `quad(epsrel=1e-4)` (quad's actual accuracy on smooth polynomials is ~1e-12).

**4. landing**: prev = sample + Σ_c coeffs[c]·derivs[−(c+1)] — **coeff[0] pairs with the newest derivative** (the official `zip(lms_coeffs, reversed(self.derivatives))`).

**Return dict**: `prev_sample` / `pred_original_sample` / `derivative` (float32), `lms_coefficients` (list of floats), `order_used` (the actual order of this call), and `state` = {`derivatives`: the history} (float64 tensors). The pre-call history-length contract is min(step_index, order) and the post-call length is min(step_index + 1, order) — the LMS history consists of real derivatives from the very first step, with **no None head**. Validation failures always raise `ValueError`.

## Example

```
sigmas = [14, 7, 3, 0.8, 0.1, 0], order = 4

step 0 (state = None): mo = 0.4, sample = 2.0
  x0 = 2.0 - 14*0.4 = -3.6, derivative = (2.0 - (-3.6))/14 = 0.4
  order_used = min(0+1, 4) = 1
  coef[0] = integral over [14, 7] of 1 dtau = 7 - 14 = -7
  prev = 2.0 + (-7)*0.4 = -0.8          (first step degenerates to Euler)
  state = {derivatives: [0.4]}

step 1 (state from step 0): mo = -0.1, sample = -0.8
  x0 = -0.8 - 7*(-0.1) = -0.1, derivative = (-0.8 - (-0.1))/7 = 0.1
  order_used = min(1+1, 4) = 2, history = [0.4, 0.1]
  nodes = (sigma[1], sigma[0]) = (7, 14)
  l_0(tau) = (tau - 14)/(7 - 14), l_1(tau) = (tau - 7)/(14 - 7)
  coef[0] = integral over [7, 3] of l_0 = -36/7, coef[1] = integral of l_1 = 8/7
  (partition of unity: coef[0] + coef[1] = sigma[2] - sigma[1] = -4)
  prev = -0.8 + (-36/7)*0.1 + (8/7)*0.4 = -0.8 - 0.5143 + 0.4571 = -0.8571
  state = {derivatives: [0.4, 0.1]}
```
