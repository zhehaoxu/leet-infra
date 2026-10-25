# Solution Notes · Speculative Decoding Rejection Sampling

## Why the distributions are equivalent

Standard sampling draws a token from the target $p$. Here we instead draw $x_t\sim q$ first and then accept with $\min(1, p/q)$. If accepted we use $x_t$ directly; if rejected we redraw from $(p-q)_+$. For any token $v$:

$$P(\text{final}=v) = \underbrace{\min(p_v,q_v)}_{\text{direct accept}} + \underbrace{(1-\sum_u \min(p_u,q_u))\cdot \frac{(p_v-q_v)_+}{\sum_w (p_w-q_w)_+}}_{\text{resample after rejection}}$$

Using the identity $\sum(p-q)_+ = \sum(p-q)_- = 1-\sum\min(p,q)$, this simplifies back to $p_v$. This is exactly why "rejection sampling correction" guarantees a distribution **exactly identical** to token-by-token sampling from the target.

## Where the bonus token comes from

When the target model verifies the $\gamma$ draft tokens, its single forward pass actually yields the distributions of $\gamma+1$ positions (the last one being "if the first $\gamma$ are all correct, what comes next"). So when everything is accepted, one extra token is produced for free — that is the source of the speedup.

## Determinism of CDF inversion

The problem requires "smallest $k$ such that $\mathrm{cumsum}[k]\ge u$" rather than `torch.multinomial`, so that given $u$ the sampling result is **reproducible** and the oracle can cross-check independently with a pure-Python accumulation loop. Watch the boundary: the last entry of `torch.cumsum` may be slightly below 1 due to floating point, but for $u\in[0,1)$ `k` will not run past $V$.
