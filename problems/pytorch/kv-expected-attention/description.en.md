# ExpectedAttention: Taking the Expectation over Future Attention

## Problem

**ExpectedAttention** (Mülbauer et al., 2024, [arXiv:2406.11430](https://arxiv.org/abs/2406.11430)): what really matters is not "how much attention past queries paid", but "**how much attention future queries are expected to give the current key**". Fit the past queries with a Gaussian model $\mathcal{N}(\mu, \Sigma)$, then compute $\mathbb{E}_q[e^{q^\top k / \sqrt{d}}]$ for the current key $k$. The Gaussian integral gives a closed form:

$$ \mathbb{E}\big[A(q, k)\big] = \exp\!\Big(\frac{\mu^\top k}{\sqrt{d}} + \frac{k^\top \Sigma\, k}{2 d}\Big) $$

Then normalize with softmax, multiply by the value's norm (**a value carrying more information is worth keeping more**), and place a "global max + 1" at the sink positions to force their retention.

## Rules

Given:
- `mu` $(H, D)$: the query mean (**after the average RoPE rotation has been applied**; this problem assumes it is provided externally)
- `cov` $(H, D, D)$: the query covariance (same as above)
- `keys` $(H, L, D)$, `values` $(H, L, D)$
- `n_sink` $s$: the first $s$ positions are not scored at all and are safeguarded with the maximum value
- `epsilon` $\varepsilon$: the offset applied before vnorm

For each head $h$:

1. Drop the sinks: $k' = \text{keys}_{h, s:}$, $v' = \text{values}_{h, s:}$
2. Unnormalized expected score:
   $$ u_{h,n} = \frac{\mu_h^\top k'_n}{\sqrt{D}} + \frac{k_n'^\top \Sigma_h\, k'_n}{2D} $$
3. Softmax along $n$: $p_{h,n} = \text{softmax}_n(u_{h,n})$
4. Scale by the value norm: $r_{h,n} = (p_{h,n} + \varepsilon)\cdot \|v'_n\|_2$
5. Fill the first $s$ positions with $m + 1$ ($m$ is the global maximum of $r$, **across heads and across positions**)

Return shape $(H, L)$.

## Examples

```
H=2, L=6, D=3, n_sink=2, epsilon=0.1

mu  = [[0.5, 0.2, -0.1], [0.1, 0.4, 0.3]]
cov = [[[1.0,0.1,0.0],[0.1,0.8,0.05],[0.0,0.05,0.6]],
       [[0.7,0.0,0.2],[0.0,0.9,0.1],[0.2,0.1,0.5]]]
keys[h=0] = [[1,0,0],[0,1,0],[0.5,0.5,0],[1,1,1],[0.2,0.3,0.4],[0,0,2]]
keys[h=1] = [[0.5,0.5,0.5],[1,0,0],[0,1,0],[2,1,0.5],[0.3,0.3,0.3],[1.5,0,0]]
values[h=0] = [[0.1,0.2,0.3],[0.4,0.5,0.6],[1,0,0],[0,1,0],[1,1,0],[0.5,0.5,0.5]]
values[h=1] = [[0.2,0.1,0],[0.3,0.3,0.3],[2,0,0],[1,1,1],[0,2,0],[0.5,1,1.5]]

softmax over n=4 keys → multiply by ||v|| → global m=0.930834
scores[h=0] = [1.930834, 1.930834, 0.322050, 0.470150, 0.404386, 0.278735]
scores[h=1] = [1.930834, 1.930834, 0.603884, 0.930834, 0.530053, 0.553043]
```

## Constraints

- CPU float32; `use_covariance = True` and `use_vnorm = True` (fixed by the problem).
- `softmax` runs along the last dimension (`n`); no normalization across heads.
- The sink positions are filled with `scores.max() + 1` — the global max across heads.
