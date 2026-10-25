# In Depth: Linear / Cosine Noise Schedule ᾱ Sequence

## Source
- linear: Ho et al., DDPM (arXiv:2006.11239), `betas = linspace(1e-4, 0.02, T)`.
- cosine: Nichol & Dhariwal, *Improved DDPM* (arXiv:2102.09672), Equation (5); a cosine-shaped \(\bar\alpha\) prevents the high-frequency range (near \(t=T\)) from being crushed to 0 too early as in linear. The engineering implementation (diffusers `cosine_beta_schedule`) derives \(\beta\) by differencing the raw cumulative product \(\tilde a_t=f(t)/f(0)\) and then applies `clip(0, 0.999)`.

## Key Points and Pitfalls
1. **Difference first, then cumprod**: the closed form cosine gives is \(\tilde a_t\) (which itself looks like ᾱ), but the standard procedure converts it into \(\beta_t=1-\tilde a_t/\tilde a_{t-1}\), clips it, and **only then** takes cumprod(1-β). Without clipping the two are almost identical; differences show up only where clipping triggers — follow the problem statement exactly.
2. **\(T+1\) sample points**: `linspace(0, T, T+1)` covers \(t=0..T\) and yields \(T\) values of β; `f(0)` serves as the normalization denominator so that \(\tilde a_0=1\).
3. **Index alignment**: output `alpha_bar[t-1]` corresponds to \(\bar\alpha_t\), and the length is exactly \(T\).
4. float32: pass `dtype=torch.float32` to `linspace`.

## Cross-check Approach
The reference solution uses `linspace` plus vectorized differencing/`cumprod`. The grader's `_oracle` uses Python `math` to compute β for each t (evaluating \(f\) point by point for cosine) and accumulates sequentially with `prev *= (1-b)` (different structure, numerically equivalent), with `allclose(atol=1e-5,rtol=1e-5)`. Additional checks cover monotonic non-increase, first entry <1, and cross-checking the values of the two schedules.

## Complexity
\(O(T)\).
