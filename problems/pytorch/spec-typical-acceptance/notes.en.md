# Solution Notes · Typical Acceptance

## Approximating the probability ratio with a threshold

Modified rejection sampling requires "knowing the draft distribution $q$": both the acceptance probability $\min(1,p/q)$ and the residual $\mathrm{norm}(\max(p-q,0))$ need the two columns $p,q$ in hand at the same time. The observation behind typical acceptance is that — at low temperature and when draft and target are close — the vast majority of draft tokens fall inside the **typical set** of the target distribution ($p(x)$ well above 0). So just drop $q$ and use a fixed threshold $\varepsilon$: accept whenever $p_t(\text{draft}[t])\ge\varepsilon$, otherwise resample one token from $p_t$ on the spot and hit the brakes. No division and no residual normalization, so verification costs almost nothing.

## Comparison with rejection sampling and Medusa

| Verification scheme | Needs q | Sampling on rejection | Structure |
|---|---|---|---|
| Modified RS (155) | Yes | $\mathrm{norm}(\max(p-q,0))$ | Single sequence |
| **Typical (this problem, 178)** | **No** | **$p_t$ directly** | **Single sequence** |
| Medusa (175) | No | — | Longest branch of a candidate tree |

All three use the "accepted prefix plus one bonus/replacement" pattern; this problem is the most linear branch of them: decide position by position, stop at the first rejection.

## The boundary of the CDF inverse lookup

Sampling always uses $\mathrm{cdf}(\text{probs},u)=\min\{k:\sum_{j\le k}\text{probs}[j]>u\}$ (strictly $>$). In the implementation that means accumulating until the first index whose cumulative sum is strictly greater than $u$. The acceptance test uses $\ge$ (equality included) — these two inequality directions must agree exactly with the oracle, otherwise the results diverge when $p=\varepsilon$ or when $u$ lands on a cumulative point. In the randomized cross-check, `u_new` is deliberately drawn close to 1 yet `<0.98` to avoid the boundary where floating-point accumulation equals 1 exactly.

## Edge-case self-checks

- `check_example`: the whole draft is accepted → append the bonus from `P[γ]`.
- `check_reject_first`: at the first position $p<\varepsilon$ → output only the replacement token, length 1.
- `check_reject_mid`: rejected midway → output = accepted prefix + replacement token, and the draft after it is no longer verified.
- `check_matches_oracle`: random $P/\text{draft}/\varepsilon/u$, cross-checked against an independent pure-Python implementation.
