# Problem

**PVeRA** (Probabilistic Vector-based Random Matrix Adaptation, arXiv:2512.07703, HF PEFT v0.20.0 `tuners/pvera`) is the probabilistic version of VeRA: it keeps the extreme parameter budget of "cross-layer shared frozen random projections \(A,B\) + two tiny trainable vectors per layer", but upgrades each layer's latent adaptation from a fixed value to a **distribution** — the forward pass samples the latent representation via the reparameterization trick, training learns \(\mu/\log\sigma^2\), and inference can optionally sample for Monte Carlo confidence intervals.

The deterministic account in the official `layer.py` (anchors fully re-computable offline):

```
pre = lambda_d * (x @ A_s.T)                # (…, 2r): lambda_d's first half scales the mu path, second half the logvar path
mu, logvar = pre.chunk(2, dim=-1)           # each (…, r)
z = mu  (eval)  |  mu + eps·exp(½·logvar)  (sampling)
delta = lambda_b * (z @ B_s.T)              # no scaling term!
```

Three key conventions:

1. **\(A\) is a double bank**: the shared `pvera_A` has shape \((2r, in_{max})\) (kaiming_uniform with default \(a=0\), CPU generator seeded by `projection_prng_key`), and \(B\) has shape \((out_{max}, r)\). Each layer slices `A[:, :in]`, `B[:out]`. μ and logvar share the **front/back halves of one linear map** (\(A\)'s top half produces μ, the bottom half logvar) — not two independent linear layers.
2. **merge takes the μ half only**: `get_delta_weight` uses `lambda_d[:r]` and `A[:r]` — the merge semantics is "expected delta": \(\Delta W = (\lambda_b \cdot B_s)\,(\lambda_{d_\mu} \cdot A_\mu)\). The logvar half is ignored by both merge and the eval forward.
3. **Init quirk**: `reset_pvera_parameters` fills `lambda_d` with `d_initial` (default 0.1) and **zeros** `lambda_b` — although `update_layer` creates it with `torch.ones`, init_weights=True immediately overwrites it via reset's `zeros_`, so the training-start delta is 0 (same zero-start semantics as VeRA); with init_weights=False the reset is skipped, leaving lambda_b all ones and lambda_d as randn.

## Problem Description

## Rules

- `mode="shared_projections"`: takes `r, in_max, out_max, prng_key`, returns float64 views of \(A,B\) plus the float32 originals (official RNG account: kaiming_uniform (\(a=0\)) reproduced bit-for-bit).
- `mode="slice_pair"`: slices `A_s (2r, in)` and `B_s (out, r)` from the shared banks.
- `mode="forward_delta"`: `eps=None` takes the eval path (z=μ, deterministic); a given `eps` takes the sampling path (z = μ + eps·exp(½logvar)). Returns the adapter increment (base excluded).
- `mode="merge_delta"`: μ-half merge \(\Delta W\ (out,in)\).
- `mode="layer_setup"`: returns `{"lambda_d": (2r,) filled with d_initial, "lambda_b": (out,) all zeros, "lambda_b_uninit": (out,) all ones}` (post-reset account plus the init_weights=False λ_b branch).
- All torch float64 (\(A/B\) originals anchored in float32 then cast, matching the in-layer `.to(dtype)` semantics).

## Example

\(r=4, in=8, out=6\) (shared banks \(A(8,10), B(12,4)\)):

```
forward_delta(x, lambda_d, lambda_b, A_s, B_s)          # eval: (2,5,6)
forward_delta(x, lambda_d, lambda_b, A_s, B_s, eps)     # sample: z = mu + eps·exp(½logvar)
merge_delta(lambda_d, lambda_b, A_s, B_s) -> (6,8)      # only lambda_d[:4] and A_s[:4]
```

## Constraints

- Eval forward is merge-consistent: `forward_delta(x, …) == x @ dW.T` (identity outside the base).
- Mutating the logvar half (`lambda_d[r:]`) changes neither the eval forward nor merge — the observable branch difference only exists in the sampling path.
- An all-zero μ half ⇒ \(\Delta W \equiv 0\) (zero-start semantics).
- The sampling path takes eps as an explicit input (the judge never draws random numbers).
