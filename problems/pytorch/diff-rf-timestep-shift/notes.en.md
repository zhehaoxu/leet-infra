# In Depth: Timestep Shift for Rectified-Flow Sampling

## Origin

Esser et al., *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis* (SD3, arXiv:2403.03206) §3 gives \(t'=s\,t/(1+(s-1)t)\) plus the resolution-dependent linear interpolation of \(\mu\); Flux's `compute_empirical_mu` (`base_seq_len=256`, `max_seq_len=4096`, `base_shift=0.5`, `max_shift=1.15`) uses the same constants.

## Core points and pitfalls

1. **It is a logit-domain shift, not a multiplication of t**: \(t'=\sigma(\operatorname{logit}(t)+\mu)\). Writing \(t\cdot s\) both leaves the range and moves the endpoints; the logit form keeps \(0\to0\) and \(1\to1\) automatically.
2. **\(\mu\) versus \(s\)**: \(\mu\) can be negative and measures a perturbation, while \(s=e^{\mu}>0\) is a multiplicative scale. Be clear which one appears in a given formula (the interface accepts both but exactly one must be passed).
3. **The interpolation extrapolates linearly**: \(\mu=m\cdot\text{seq\_len}+b\) without clamping to `[base_shift, max_shift]`, so 64 tokens yield a \(\mu\) below `base_shift`. An implementation that clamps fails the short-sequence assertion.
4. **Additivity is the best unit test**: \(\sigma(\sigma(\operatorname{logit}t+\mu_1)+\mu_2)=\sigma(\operatorname{logit}t+\mu_1+\mu_2)\), so "two shifts equal one shift by the product".
5. **Inverse map**: \(t=t'/(s-(s-1)t')\), structurally the same as the forward map; samplers that invert (image-to-image, editing) must get back exactly where they started.

## Verification approach

The reference solution converts the whole grid to float64 and computes in one shot. The judge's `_oracle` takes another route: element-wise `sigmoid(log(t/(1-t)) + mu)` with the endpoints returned as 0/1 to avoid the ±inf of logit — a different source from the reference rational form. It adds four structural groups: endpoints and monotonicity, identity (\(s=1\)) and composition, three interpolation points for `seq_len` (base / max / midpoint) plus monotonicity, and the inverse-map round trip. Tolerance `allclose(atol=1e-6)`.

## Complexity

Element-wise \(O(N)\), plus one `exp` and two divisions.
