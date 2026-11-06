# DPMSolverMultistep Inverse Scheduler: Multistep Update Ledger

## Background

`DPMSolverMultistepInverseScheduler` is the reverse counterpart of `DPMSolverMultistepScheduler` in diffusers: it pushes a clean sample back to a noisier level along an **increasing** sigma table, for image editing pipelines that invert to noise, edit the noise, and sample again. Its first/second/third-order update functions are copied verbatim from the forward scheduler, but because the sigma table runs in the opposite direction, `h = lambda_t - lambda_s < 0`, `exp(-h) > 1`, and `exp(h) < 1`: every coefficient lands in a different numerical regime than in forward sampling.

This problem reconstructs only the judgeable ledger inside the official `step()` chain. The Brownian noise terms of the `sde-*` branches, dynamic thresholding, `learned_range` channel slicing, and `set_timesteps` sigma-table construction are out of scope (the first two already have `diff-cosine-dpmsolver-sde-step` and `diff-dpm-x0-conversion-threshold`).

## Rules

The `sigmas` table is 1-D, strictly increasing, and finite positive; `step_index = i`. Flow mode additionally requires `0 < sigma < 1`. All tensors are fp64.

### 1. sigma to (alpha, sigma_t) and lambda

```
VP   : alpha = 1 / sqrt(1 + sigma^2),  sigma_t = sigma * alpha
flow : alpha = 1 - sigma,             sigma_t = sigma
lam  = log(alpha) - log(sigma_t)
```

Hence `h = lam[i+1] - lam[i]` (always negative in the inverse direction); second order also needs `h_0 = lam[i] - lam[i-1]`, third order additionally `h_1 = lam[i-1] - lam[i-2]`, with `r0 = h_0/h` and `r1 = h_1/h`. Note the official code overwrites the local name `sigma_t` with the converted value, so `sigma_t/sigma_s` uses converted quantities.

### 2. convert_model_output

For `algorithm_type = "dpmsolver++"` the solver quantity is `x0_pred`; for `"dpmsolver"` it is `epsilon`. `prediction_type` selects the conversion. `sigma` is `sigmas[i]`, and `(alpha_t, sigma_t)` comes from section 1:

```
++  epsilon      : (sample - sigma_t * mo) / alpha_t
++  sample       : mo
++  v_prediction : alpha_t * sample - sigma_t * mo
++  flow_prediction : sample - sigma * mo
dpm epsilon      : mo
dpm sample       : (sample - alpha_t * mo) / sigma_t
dpm v_prediction : alpha_t * mo + sigma_t * sample
```

`flow_prediction` uses the **raw, unconverted sigma** (the official code shadows the local name), not `sigma_t`. An invalid `prediction_type` or `algorithm_type` raises `ValueError`.

### 3. First-order update

`m0` is the converted model output of the current step; `c_sample` and `c_model` are the coefficients of `sample` and `m0`, with `prev = c_sample*sample + c_model*m0`:

```
++  : c_sample = sigma_t / sigma_s,  c_model = -(alpha_t * (exp(-h) - 1))
dpm : c_sample = alpha_t / alpha_s, c_model = -(sigma_t * (exp(h) - 1))
```

### 4. Second-order update

History `model_outputs = [m1, m0]` (old to new), `D0 = m0`, `D1 = (1/r0)*(m0 - m1)`. Let `B++ = alpha_t*(exp(-h) - 1)` and `Bdpm = sigma_t*(exp(h) - 1)`:

```
++  midpoint : c_sample = sigma_t/sigma_s0, c_d0 = -B++,      c_d1 = -0.5 * B++
++  heun     : c_sample = sigma_t/sigma_s0, c_d0 = -B++,      c_d1 = B++/h + alpha_t
dpm midpoint: c_sample = alpha_t/alpha_s0, c_d0 = -Bdpm,     c_d1 = -0.5 * Bdpm
dpm heun    : c_sample = alpha_t/alpha_s0, c_d0 = -Bdpm,     c_d1 = -(Bdpm/h - sigma_t)
```

`prev = c_sample*sample + c_d0*D0 + c_d1*D1`.

### 5. Third-order update

History `[m2, m1, m0]`, `D1_0 = (1/r0)*(m0 - m1)`, `D1_1 = (1/r1)*(m1 - m2)`, `D1 = D1_0 + r0/(r0+r1)*(D1_0 - D1_1)`, `D2 = (1/(r0+r1))*(D1_0 - D1_1)`, `D0 = m0`. Third order is a fixed formula with no `solver_type`:

```
++  : c_d0 = -B++, c_d1 = B++/h + alpha_t,
      c_d2 = -(alpha_t * ((exp(-h) - 1 + h) / h^2 - 0.5))
dpm : c_d0 = -Bdpm, c_d1 = -(Bdpm/h - sigma_t),
      c_d2 = -(sigma_t * ((exp(h) - 1 - h) / h^2 - 0.5))
```

`c_sample` is as in second order; `prev = c_sample*sample + c_d0*D0 + c_d1*D1 + c_d2*D2`.

### 6. step state machine

A single call runs, in order: convert the current `model_output` -> shift history one slot left and append the current result -> compute the two lower-order flags -> select the order -> increment the counter. Note the inverse scheduler's lower-order conditions **do not** include the forward scheduler's `final_sigmas_type == "zero"` term:

```
lower_order_final  = (i == N-1) and (euler_at_final or (lower_order_final and N < 15))
lower_order_second = (i == N-2) and lower_order_final and N < 15
order 1 : solver_order == 1 or lower_order_nums < 1 or lower_order_final
order 2 : solver_order == 2 or lower_order_nums < 2 or lower_order_second
order 3 : otherwise
```

`N = n_steps = len(sigmas) - 1`; `state = None` means a fresh start (history all `None`, counter 0). On exit `lower_order_nums = min(lower_order_nums + 1, solver_order)`, and the new `state = {model_outputs, lower_order_nums}` is returned.

## Judge input and output

Single entry point `solve(mode, **kw)`. Inputs and outputs of the six modes:

- `sigma_to_alpha`: `sigma`, `use_flow_sigmas` -> `{alpha, sigma_t}`;
- `convert`: `model_output`, `sample`, `sigma`, `algorithm_type`, `prediction_type`, `use_flow_sigmas` -> `{converted, kind}`;
- `first`: `sigmas`, `step_index`, `sample`, `model_output`, `algorithm_type`, `use_flow_sigmas` -> `prev_sample` plus all intermediates;
- `second`: additionally `model_outputs` (length 2) and `solver_type` -> same shape;
- `third`: additionally `model_outputs` (length 3) -> same shape;
- `step`: `sigmas`, `step_index`, `n_steps`, `sample`, `model_output`, `state` and all config -> `prev_sample`, `used_order`, `kind`, `converted`, both lower-order flags, counter before/after, `state`, and that order's intermediates.

## Example

Flow mode, `sigmas = [0.2, 0.4]`, `step_index = 0`: `lambda_s = log 4`, `lambda_t = log 1.5`, `h = log(3/8)`, `exp(-h) = 8/3`, `exp(h) = 3/8`, `alpha_t = 0.6`, `sigma_t = 0.4`, `sigma_s = 0.2`.

```
++  : c_sample = 0.4/0.2 = 2.0, c_model = -(0.6 * (8/3 - 1)) = -1.0
      prev = 2.0 * 1.0 - 1.0 * 0.5 = 1.5
dpm : c_sample = 0.6/0.8 = 0.75, c_model = -(0.4 * (3/8 - 1)) = 0.25
      prev = 0.75 * 1.0 + 0.25 * 0.5 = 0.875
```

## Constraints

- `sigmas` strictly increasing, finite positive; flow mode needs `0 < sigma < 1`; `step_index` and the order must leave enough preceding sigmas, otherwise `ValueError`.
- `algorithm_type in {dpmsolver++, dpmsolver}`, `solver_type in {midpoint, heun}`, `solver_order in {1,2,3}`, `prediction_type` per section 2; anything else raises `ValueError`.
- No randomness; tensors are computed elementwise, shapes follow the inputs, results are fp64.
