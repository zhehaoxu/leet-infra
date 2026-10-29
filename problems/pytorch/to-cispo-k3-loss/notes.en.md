# Solution Notes · CISPO and K3

## Moving the clip from the objective to the weight

PPO's $\min(\rho A,\mathrm{clip}(\rho)A)$ has an often-overlooked consequence: when $\rho$ is out of range and `min` selects the constant term, that token's gradient **becomes exactly 0**. In long-sequence RL, once the out-of-range fraction rises, a substantial share of tokens is sampled for nothing. CISPO treats the clipped result as "each token's importance weight": the weight is bounded while `logπ` is still differentiated for every token — the gradient never vanishes, only its magnitude is capped. This does not conflict with the trust-region intent: what is capped is precisely a single token's contribution to the gradient.

## Why stop-gradient is mandatory

CISPO's weight $w=\mathrm{clamp}(\rho)$ depends on $\rho$, which depends on $\log\pi$. Without `detach`, differentiating w.r.t. $\log\pi$ adds a $\partial w/\partial\log\pi=\rho$ term (in the unclipped region), turning the objective into "weighted logπ plus the weight's own variation", which is not what the paper defines. With stop-gradient, $\partial L/\partial\log\pi_t=-w_tA/N$ is clean and analytically checkable per token — the test cross-checks it against autograd exactly.

## Why K3 is non-negative and direction-sensitive

Let $r=\pi_\text{ref}/\pi$, so $k_3=r-\log r-1$. With $f(r)=r-\log r-1$ and $f'(r)=1-1/r$, the minimum is at $r=1$ with $f(1)=0$, hence $k_3\ge0$ for any $r>0$. Its expectation is $\mathrm{KL}(\pi\|\pi_\text{ref})$ because $\mathbb{E}_\pi[r]=1$ and $\mathbb{E}_\pi[\log r]=\mathrm{KL}(\pi\|\pi_\text{ref})$. The direction cannot be reversed — samples come from the **current** policy, so the ratio must be $\pi_\text{ref}/\pi$; writing $\pi/\pi_\text{ref}$ yields another non-negative quantity estimating the reverse KL, numerically entirely different. A single asymmetric sample separates the two directions in the tests.

## Why the overflow guard sits at 20

The $e^{\text{log\_r}}$ in $k_3$ overflows for large $\text{log\_r}$. float32's exponent limit is about 88 and float64's about 709, but $k_3$ is already meaninglessly large beyond $\text{log\_r}>20$ ($e^{20}\approx4.85\times10^8$), and a real logprob difference of that magnitude usually signals a data or masking bug. The contract therefore errors at 20, turning "silently compute inf and contaminate the batch" into an explicit input error.

## Clip fraction as a diagnostic

`clip_frac` measures the out-of-range token share — a cheap indicator of "how far the policy has drifted / whether the clipping radii are right". It uses **strict** inequalities: $\rho$ exactly equal to $1+\epsilon_\text{high}$ is not out of range, because `clamp` changes nothing and no behavior differs. This small convention makes "boundary" and "out of range" mutually exclusive in the statistic, so implementations using `>=` do not miscount boundary tokens.
