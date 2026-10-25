# Speculative Decoding: Acceptance / Rejection Criterion

## Problem

**Speculative Decoding** (Leviathan et al. 2022) uses a cheap small draft model to guess $\gamma$ tokens at once, then uses the large target model to **verify all $\gamma$ guesses in parallel with a single forward pass**. To guarantee that the final sampling distribution is exactly the target distribution $p$, **rejection sampling** is used as a correction: for each draft token $x_t$, accept it with probability $\min\!\big(1, \frac{p(x_t)}{q(x_t)}\big)$ ($q$ is the draft distribution); as soon as some step is rejected, discard that step and all later guesses and instead resample a token from the **corrected distribution** $\mathrm{normalize}(\max(0,\,p-q))$ to finish. If all $\gamma$ are accepted, one extra **bonus token** is taken for free from the target's distribution at step $\gamma$ (since the target already computed $\gamma+1$ positions anyway).

This way each decoding step produces at most $\gamma+1$ tokens, with a distribution **exactly equivalent** to token-by-token sampling, while greatly reducing the number of serial forward passes of the target.

## Rules

Given `draft_probs` $(\gamma, V)$, `target_probs` $(\gamma+1, V)$, `draft_tokens` $(\gamma,)$, `u_accept` $(\gamma,)$, and the scalar `u_resample`:

1. Iterate $i = 0 \ldots \gamma-1$ in order and take the draft guess $x_t = $`draft_tokens[i]`:
   - Acceptance probability $\mathrm{ratio} = \dfrac{p_i[x_t]}{q_i[x_t]}$; if $q_i[x_t]=0$, treat it as $\mathrm{ratio}=+\infty$
   - If `u_accept[i]` $< \min(1, \mathrm{ratio})$: **accept**, output $x_t$, and proceed to $i+1$
   - Otherwise: **reject**, let the corrected distribution be $\pi = \mathrm{normalize}\big(\max(0,\; p_i - q_i)\big)$ (element-wise: clamp to $\ge 0$ first, then divide by its sum), sample one token from $\pi$ and append it to the output, then **stop immediately** (all later drafts are discarded)
2. If the loop finishes without rejection (all $\gamma$ accepted): sample one token from `target_probs[γ]` (the bonus-step distribution) and append it to the output
3. **Sampling operator** (given a distribution $\pi$ and a uniform number $u$): return the **smallest** index $k$ such that $\sum_{j \le k} \pi[j] \ge u$ (i.e. CDF inversion, with `>=` for the floating-point comparison)

Return a 1-D `torch.long` tensor: the token id sequence output by this decoding step (length $\le \gamma+1$).

## Examples

```
γ=3, V=4
draft_probs  = [[0.5,0.3,0.1,0.1],
                [0.2,0.5,0.2,0.1],
                [0.1,0.1,0.6,0.2]]
target_probs = [[0.3,0.4,0.2,0.1],
                [0.4,0.2,0.3,0.1],
                [0.1,0.1,0.6,0.2],
                [0.25,0.25,0.25,0.25]]
draft_tokens = [0, 1, 2]
u_accept     = [0.5, 0.7, 0.9]
u_resample   = 0.5

i=0: x_t=0, ratio=0.3/0.5=0.6, u=0.5<0.6 → accept tok0
i=1: x_t=1, ratio=0.2/0.5=0.4, u=0.7≥0.4 → reject
     correction = clamp([0.4-0.2, 0.2-0.5, 0.3-0.2, 0.1-0.1],0)=[0.2,0,0.1,0]
     normalize → [0.6667,0,0.3333,0], CDF=[0.6667,0.6667,1.0,1.0]
     u=0.5 → smallest k with CDF≥0.5 → k=0
Output = [0, 0]   (tok0 accepted, then the rejection position resamples to 0; i=2 is discarded)
```

## Constraints

- CPU float32; every row of all distributions sums to 1.
- The sampling index uses CDF inversion (`>=` comparison), not `torch.multinomial` (to guarantee determinism).
- The output is a 1-D `torch.long` tensor.
