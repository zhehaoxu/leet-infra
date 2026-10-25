# In Depth: One Deterministic DDIM Denoising Step

## Origin
Song et al., *Denoising Diffusion Implicit Models* (arXiv:2010.02502), Eq. (12): \(x_{t-1}=\sqrt{\bar\alpha_{t-1}}\hat x_0+\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\,\epsilon_\theta+\sigma_t z\). Setting \(\sigma_t=0\) gives this problem's deterministic update (\(\epsilon_\theta\) is the model-predicted noise, and \(\hat x_0\) is inverted from it).

## Core points and pitfalls
1. **Invert \(\hat x_0\) first, then jump forward**: \(\hat x_0=(x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta)/\sqrt{\bar\alpha_t}\), then recombine according to \(\bar\alpha_{t-1}\). Neither step can be omitted, and the direction cannot be reversed.
2. **\(\sigma_t=0\) means no noise**: unlike the DDPM posterior, the deterministic DDIM version injects no \(z\), so the same input always yields the same output — which is why it supports few-step sampling and "inversion".
3. **Identity / full-denoise degeneracies**: \(\bar\alpha_{t-1}=\bar\alpha_t\) should recover \(x_t\); \(\bar\alpha_{t-1}=1\) should equal \(\hat x_0\). These two are probes for whether the formula is written correctly.
4. The denominator is \(\sqrt{\bar\alpha_t}\), and \(\bar\alpha_t\in(0,1]\) is never zero.

## Verification approach
The reference solution vectorizes the two steps. The judge's `_oracle` recomputes \(\hat x_0\) and \(x_{t-1}\) with `math.sqrt` using a doubly nested element-wise loop; it also uses "forward generation → DDIM inversion" to verify that \(\hat x_0\) is recovered, that \(\bar\alpha_{t-1}=\bar\alpha_t\) is the identity, and that \(\bar\alpha_{t-1}=1\) returns \(\hat x_0\) (different structure, numerically equivalent), with `allclose(atol=1e-5,rtol=1e-5)`.

## Complexity
Element-wise \(O(N)\).
