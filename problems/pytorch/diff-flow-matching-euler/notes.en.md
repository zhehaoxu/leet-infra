# In Depth: Euler Integration of a Rectified-Flow Velocity Field

## Origin

Lipman et al., *Flow Matching for Generative Modeling* (arXiv:2210.02747) give the conditional flow-matching objective; Liu et al., *Rectified Flow* (arXiv:2209.14577) use the straight path \(x_t=(1-t)x_0+t\epsilon\) with \(v=\epsilon-x_0\) to turn sampling into a monotone ODE; Esser et al., *Scaling Rectified Flow Transformers* (SD3, arXiv:2403.03206) scale it into a latent DiT. The SD3 / Flux sampler is exactly this Euler step.

## Core points and pitfalls

1. **Direction of time**: training uses \(t=0\) for data and \(t=1\) for noise, but sampling runs \(t:1\to0\), hence \(\Delta t<0\). Reversing the direction without flipping the velocity produces a sampler that gets noisier.
2. **Definition of the velocity**: it is \(v=\epsilon-x_0\) (pointing at the noise), not \(x_0-\epsilon\). On the straight path it is constant, which is exactly why first-order Euler is exact for RF; mixing in the ε-prediction "v" merges two different conventions.
3. **Trajectory entries must be fresh tensors**: storing a state in a list and then doing in-place `x += ...` corrupts the history; the judge catches it with "the first entry still equals the start".
4. **Return semantics**: `x_final` must carry the same value as `trajectory[-1]` and the list length is exactly `n_steps + 1`.
5. **Scalar velocities**: `velocity_fn` may return a python float (e.g. \(v\equiv c\)); convert it to a tensor and broadcast instead of multiplying `None` or raising.

## Verification approach

The reference solution accumulates in float64 and stores states as float32. The judge's `_oracle` first materializes the whole time grid and then adds/multiplies element by element in python floats (a different source from the reference tensor ops), adding four structural groups: the grid and the call count (the field is invoked exactly `n_steps` times), the constant-velocity closed form \(x_{\text{init}}+(t_{\text{end}}-t_{\text{start}})c\), the RF straight path with 1/2/7 steps recovering \(x_0\) while every intermediate state stays on the line, and a zero velocity leaving the trajectory in place. Tolerance `allclose(atol=1e-5)`.

## Complexity

\(O(\text{steps})\) velocity-field calls; each call costs whatever `velocity_fn` costs, while the state updates cost \(O(\text{steps}\cdot N)\) memory traffic.
