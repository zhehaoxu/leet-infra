# Typical Acceptance: Thresholding Target Probabilities for Speculative Verification

## Problem

The rejection sampling of standard speculative decoding needs the draft distribution $q$ to compute the acceptance probability $\min(1,p/q)$, and on rejection it must resample from the residual $\mathrm{norm}(\max(p-q,0))$ — two distributions to maintain. **Typical acceptance** (Leviathan et al., 2023, [arXiv:2211.17192](https://arxiv.org/abs/2211.17192) §3.2) offers a simpler approximation: use only the target distribution $p$ and treat a draft token $x$ as acceptable when it lies inside the **typical set** (i.e. $p(x)$ is no smaller than the threshold $\varepsilon$). It decides token by token over a single draft sequence; as soon as some token is judged rejected, it samples one token from the current target distribution $p_t$ as a replacement and stops.

## Rules

Arguments:
- `P`: target probabilities of shape $(\gamma+1, V)$; `P[t]` is the distribution at position $t$; the last row `P[γ]` is used for the bonus sample.
- `draft`: a sequence of $\gamma$ speculative token ids.
- `eps`: the acceptance threshold $\varepsilon$.
- `u_new`: a uniform number in $[0,1)$, used for the replacement sample at the rejection position and for the bonus sample when everything is accepted (CDF inverse lookup).

For $t=0,1,\dots,\gamma-1$:

1. Take $x=\text{draft}[t]$. If $P[t][x]\ge \varepsilon$, **accept** $x$, append it to the output, and continue to the next position.
2. Otherwise **reject**: sample one token from $P[t]$ via $\mathrm{cdf}(\cdot, u_{\text{new}})$, append it, and **stop immediately**.

If all $\gamma$ draft tokens are accepted, sample one bonus token from $P[\gamma]$ via $\mathrm{cdf}(\cdot, u_{\text{new}})$ and append it.

Return a one-dimensional `torch.long` tensor (the accepted prefix plus the replacement token at the rejection position, or everything plus the bonus).

Here $\mathrm{cdf}(\text{probs}, u)$ returns the **smallest** $k$ such that $\sum_{j\le k}\text{probs}[j] > u$.

- Unlike modified rejection sampling ([arXiv:2211.17192](https://arxiv.org/abs/2211.17192) §3.1 / [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)), typical acceptance does **not** need the draft distribution $q$, and on rejection it samples $p_t$ directly instead of the residual distribution — an "approximately lossless" simplified verification.
- It is not the same thing as `typical-sampling` in the sampling collection (a top-p style cut based on entropy/perplexity): here it is the draft verification criterion of speculative decoding.

## Examples

```
P = [[0.9, 0.05, 0.05],
     [0.1, 0.2,  0.7 ],
     [0.3, 0.3,  0.4 ],
     [0.2, 0.6,  0.2 ]]      # gamma=3, V=3
draft=[0,0,2], eps=0.1, u_new=0.4

t0: P[0][0]=0.9 >=0.1 accepted
t1: P[1][0]=0.1 >=0.1 accepted
t2: P[2][2]=0.4 >=0.1 accepted
all accepted -> bonus = cdf(P[3]=[0.2,0.6,0.2], 0.4): cum=[0.2,0.8,..] -> index 1

Output = [0, 0, 2, 1]
```

## Constraints

- CPU float32; return a one-dimensional `torch.long` tensor, compared with `torch.equal`.
- The acceptance criterion is $P[t][x]\ge\varepsilon$ (equality included).
- On rejection, sample from $P[t]$ and stop; do not keep verifying the later draft tokens.
- For sampling boundaries always take the **smallest** $k$ whose cumulative sum is strictly $>u$, matching the standalone oracle.
