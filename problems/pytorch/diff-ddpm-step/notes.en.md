# In Depth: One DDPM Reverse Posterior Step p(x_{t-1}|x_t,x_0)

## Origin
Ho et al., DDPM (arXiv:2006.11239), Eq. (7)–(11). Writing \(q(x_{t-1}|x_t,x_0)\) as a linear combination of \(x_t,\hat x_0\), both the mean coefficients and the variance \(\tilde\beta_t\) are closed-form; at sampling time the model-predicted \(\hat x_0=x_\theta\)-derived result replaces the true \(x_0\).

## Core points and pitfalls
1. **Don't confuse the three scalars**: \(\beta_t\) (per-step residual variance), \(\bar\alpha_t\) (cumulative signal), \(\alpha_t=1-\beta_t\) (per-step signal). The denominator is always \(1-\bar\alpha_t\).
2. **Which coefficient goes with which term**: in front of \(x_t\) is \(\sqrt{\alpha_t}(1-\bar\alpha_{t-1})/(1-\bar\alpha_t)\); in front of \(\hat x_0\) is \(\sqrt{\bar\alpha_{t-1}}\beta_t/(1-\bar\alpha_t)\). Swapping them yields a wrong trajectory.
3. **The variance is \(\tilde\beta_t\)**: \(\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t\) (not \(\beta_t\) itself, nor \(\bar\beta_t\)). For the last step \(t=1\), it is customary to set \(\bar\alpha_0=1\) so that the variance is 0.
4. \(z\) is passed in explicitly for determinism; \(z=0\) gives only the mean.

## Verification approach
The reference solution vectorizes: after computing \(c_t,c_0,\text{var}\), it evaluates `c_t*x_t + c_0*x0p + sqrt(var)*z`. The judge's `_oracle` recomputes the same expression with `math.sqrt` using a doubly nested element-wise loop, and asserts that for \(z=0\) the output equals "the mean with the noise term removed" (different structure, numerically equivalent), with `allclose(atol=1e-5,rtol=1e-5)`.

## Complexity
Element-wise \(O(N)\).
