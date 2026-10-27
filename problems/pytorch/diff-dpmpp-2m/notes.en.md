# In Depth: DPM-Solver++(2M) Second-Order Update

## Origin

Lu et al., *DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models* (arXiv:2211.01095) introduce the data-prediction form and the multistep second order; diffusers' `DPMSolverMultistepScheduler` (`algorithm_type="dpmsolver++"`) implements exactly this, with `solver_order=1` degenerating to Euler.

## Core points and pitfalls

1. **λ is half the log-SNR**: \(\lambda=\log(\alpha/\sigma)\). Using `log(α/σ)` and `log(ᾱ/(1-ᾱ))/2` is numerically equivalent, but do not mix it up with `log(ᾱ/(1-ᾱ))` — a factor of two throws the whole exponential term off.
2. **Sign of the coefficient**: implementations write \(\alpha_t(1-e^{-h})\), identical to \(- \alpha_t(e^{-h}-1)\); writing \(1-e^{h}\) diverges. Since \(h>0\), \(1-e^{-h}\in(0,1)\) and the coefficient is always positive.
3. **First order *is* DDIM**: \(\alpha_tD_0+\sigma_t\,(x-\alpha_{s_0}D_0)/\sigma_{s_0}\) is algebraically the same step, and is the strongest probe for whether the coefficients are right.
4. **Second order only adds 0.5·D1**: since \(D_1=(D_0-D_0^{\text{prev}})/r_0\), the identity "second order minus first order = coefficient·0.5·D1" is verifiable on its own; with \(D_1=0\) the step must degenerate.
5. **Direction of r0**: \(r_0=\lambda_{s_0}-\lambda_{s_1}>0\) requires \(s_1\) to be noisier. Flipping it flips the sign of the whole correction and the sampler collapses.

## Verification approach

The reference solution vectorizes in float64. The judge's `_oracle` recomputes element by element using the sigmoid form of λ (`0.5*log(ab/(1-ab))`), a different source from the reference `log(α/σ)`, and adds four structural groups: bitwise agreement with the DDIM closed form, the second-order difference equalling \(coef\cdot 0.5\cdot D_1\), degeneration when both predictions match, and the \(h=0\) identity plus the \(\bar\alpha_t\to1\) limit (probed with \(\bar\alpha_t=0.9999999999\)). Tolerance `allclose(atol=1e-5)`.

## Complexity

Element-wise \(O(N)\); a full multi-step trajectory costs \(O(\text{steps}\cdot N)\).
