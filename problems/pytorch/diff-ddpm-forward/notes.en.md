# In Depth: DDPM Forward Noising q(x_t|x_0)

## Origin
Ho et al., *Denoising Diffusion Probabilistic Models* (arXiv:2006.11239), Eq. (5)-(6): the Gaussian closure of \(q(x_1|x_0)*\dots*q(x_t|x_{t-1})\) yields the directly samplable \(q(x_t|x_0)=\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)\), hence \(x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon\).

## Core points and pitfalls
1. **Don't swap the two sqrt terms**: the signal coefficient is \(\sqrt{\bar\alpha_t}\) and the noise coefficient is \(\sqrt{1-\bar\alpha_t}\); their squares sum to 1 (variance preserving). Swapping them makes noise dominate when \(\bar\alpha_t\) is large.
2. **eps is passed in explicitly**: during training \(\epsilon\sim\mathcal N(0,I)\) is random, but grading must be deterministic, so the noise is an input argument and this problem performs no internal sampling.
3. **One shot**: there is no need to iterate \(t\) single-step noising operations; using \(\bar\alpha_t\) in closed form is precisely the key to DDPM's efficient training.
4. Any shape is supported; element-wise is enough.

## Verification approach
The reference solution is `sqrt(ab)*x0 + sqrt(1-ab)*eps`. The judge's `_oracle` uses Python `math.sqrt` for the scalar coefficients and then multiplies and adds element-wise, and asserts \(x_t=x_0\) at the \(\bar\alpha=1\) boundary (different structure, numerically equivalent), with `allclose(atol=1e-5,rtol=1e-5)`.

## Complexity
Element-wise \(O(N)\).
