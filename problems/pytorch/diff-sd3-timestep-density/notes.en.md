# Solution Notes · SD3 Timestep-Sampling Density

## Source

- `huggingface/diffusers`, `src/diffusers/trainers/training_utils.py`: `compute_density_for_timestep_sampling(weighting_scheme, batch_size, logit_mean, logit_std, mode_scale, ...)`. Its three branches are `logit_normal` (`torch.normal` followed by `sigmoid`), `mode` (uniform draw passed through a cosine transform), and the fallback (`torch.rand`). This problem keeps only the pure part and takes the draws as inputs.
- Esser et al., *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis* (SD3, arXiv:2403.03206) introduces this density design and `mode` sampling; the diffusers implementation came in via PR #8528 (the source docstring credits Rafie Walker).
- The function returns a **continuous position** in \([0,1]\); training scripts multiply it by `num_train_timesteps` and cast to an integer. It is not an integer timestep.

## Core Points and Pitfalls

**`logit_normal` draws a normal, not a uniform.** The "logit" in the name comes from pushing a normal sample through a sigmoid to obtain a logit-normal distribution; it does not mean sampling the logit uniformly. Feeding \(U(0,1)\) into the sigmoid gives a completely different density. The real implementation draws with `torch.normal`, and here \(z\) is supplied by the caller.

**Signs and bracketing in `mode`.** The formula is `1 - u - s * (cos²(πu/2) - 1 + u)`: start from `1-u`, then subtract `s` times the whole bracket, where the bracket is `cos² - 1 + u`. Three common mistakes: writing `u` instead of `1-u`; using `cos` instead of `cos²`; dropping `-1+u` and keeping only `cos²`. The fact that `u=0.5` must give exactly `0.5` is a quick check that the bracket is right.

**`mode_scale=0` is not the identity.** It degenerates to `1-u` (a flipped uniform), not `u`; only `uniform` is the identity. The tests pin this down.

**Endpoints and symmetry.** For any \(s\ge0\): \(v(0)=1\), \(v(1)=0\), \(v(0.5)=0.5\), and \(v(u)+v(1-u)=1\). The first three follow from the endpoint values of the cosine term, the last from \(\cos^2(\pi u/2)+\sin^2(\pi u/2)=1\). None of them depend on \(s\), so they make good property tests.

**The `logit_std=0` degenerate case.** The whole batch equals \(\sigma(\mu)\) and strict monotonicity no longer holds; test it separately rather than folding it into the "strictly increasing in z" assertion.

**Fallback behavior differs on purpose.** The source silently routes any unknown string to the uniform branch; this problem narrows the contract to three legal names and raises `ValueError` otherwise, so typos are not swallowed. This is an intentional scoping change relative to the source.

**Precision.** Compute in float64 and cast to float32 at the output; chained `cos²` and `exp` drift at the 1e-7 level in float32 versus the float64 oracle, so property tests use a 1e-6 tolerance.

## Division of Labor

- `diff-rf-timestep-shift`: given a timestep, applies a one-parameter shift toward the noisy side; this problem answers "where do timesteps come from".
- `diff-zero-terminal-snr`: modifies the endpoints of the noise schedule's \(\bar\alpha_t\); here only the sampling density changes, not the schedule.
- `diff-ddim`, `diff-ddpm-forward`: samplers / forward process; this is the training-side sampling policy.
- `diff-min-snr-weight`: the neighboring bookkeeping in the same training script — after drawing the timestep, weight the loss by SNR; together the two problems cover the SD3 timestep pipeline.
