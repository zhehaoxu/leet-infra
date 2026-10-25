# In Depth: Classifier-Free Guidance (CFG)

## Origin
Ho & Salimans, *Classifier-Free Diffusion Guidance* (arXiv:2207.12598). During training the condition is "dropped" with some probability, so that a single network can produce both \(\epsilon_\theta(x_t,t,y)\) (conditional) and \(\epsilon_\theta(x_t,t,\varnothing)\) (unconditional); at sampling time they are extrapolated linearly by weight. Engineering implementations (diffusers `StableDiffusion`) use `noise = noise_uncond + guidance_scale*(noise_cond - noise_uncond)`, and this problem adopts that convention.

## Core points and pitfalls
1. **Parameterization difference**: diffusers' `guidance_scale=w` differs from the paper's weight \(w'\) by 1: \(w=w'+1\). This problem uniformly uses \(\epsilon_u+w(\epsilon_c-\epsilon_u)\), where \(w=1\) means "no guidance, pure conditional".
2. **The direction is cond−uncond**: extrapolation follows the "conditional minus unconditional" direction; reversing it pushes samples toward the unconditional.
3. **Equivalent form**: \(\epsilon_u+w(\epsilon_c-\epsilon_u)=(1-w)\epsilon_u+w\epsilon_c\); the two formulations are numerically identical and can serve as a self-check.
4. The result may overflow the input range (normal when \(w\) is large); no clipping is performed.

## Verification approach
The reference solution is `eps_uncond + w*(eps_cond-eps_uncond)`. The judge's `_oracle` recomputes element-wise with the equivalent expression `(1-w)*eps_uncond + w*eps_cond` (different structure, numerically equivalent), using `allclose(atol=1e-5,rtol=1e-5)`. Two degenerate assertions are added: \(w=0\to\epsilon_u\) and \(w=1\to\epsilon_c\).

## Complexity
Element-wise \(O(N)\).
