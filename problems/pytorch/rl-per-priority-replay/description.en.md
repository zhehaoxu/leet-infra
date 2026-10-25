# RL Systems: Prioritized Replay Sampling and Importance-Sampling Weights

## Problem

A replay buffer holds tens of thousands of transitions but each training step only consumes $K$ of them — **which** $K$ get picked and **how much weight** each carries is a data-flow/infrastructure question, not an algorithmic one. **PER** (Prioritized Experience Replay, Schaul et al., arXiv:1511.05952) replaces uniform sampling with proportional-to-priority sampling

$$P(i)=\frac{p_i^{\alpha}}{\sum_j p_j^{\alpha}}$$

where $\alpha=0$ falls back to uniform and $\alpha=1$ is fully proportional. The resulting distribution bias is corrected with **importance-sampling (IS) weights**:

$$\tilde w_i=\bigl(N\cdot P(i)\bigr)^{-\beta},\qquad w_i=\frac{\tilde w_i}{\max_j \tilde w_j}$$

$\beta=0$ applies no correction, $\beta=1$ applies the full correction (in practice $\beta$ is annealed from 0 to 1); dividing by the batch maximum is the standard normalization and keeps every weight $\le 1$.

> On the sign of the exponent: the paper's formula is quoted both ways across the literature. **This problem fixes the mainstream code convention $(N\cdot P(i))^{-\beta}$** — the $\beta$-power of "uniform density / actual density" — so do not flip the sign yourself.

**Randomness is an explicit input**: no dice may be rolled inside the function. The uniform $[0,1)$ draws are handed to you, and what you must reproduce is the deterministic chain "which samples these draws land on, and with what weights" — the other half of PER's math, next to the sum-tree.

## Rules

Four inputs:

| Name | Type | Meaning |
|---|---|---|
| `priorities` | float $(N,)$ | priority $p_i\ge 0$ of each buffered sample (typically $\lvert\mathrm{TD}\rvert+\varepsilon$) |
| `uniform_draws` | float $(K,)$ | externally drawn uniform numbers $u_k\in[0,1)$ |
| `alpha` | scalar $\ge 0$ | priority exponent |
| `beta` | scalar $\ge 0$ | IS correction exponent |

Three computation steps:

1. **Sampling probabilities** $P(i)=p_i^{\alpha}/\sum_j p_j^{\alpha}$. If $\sum_j p_j^{\alpha}=0$ (all priorities zero with $\alpha>0$), degenerate to uniform $P(i)=1/N$.
2. **Which sample is hit**: with prefix sums $F_i=\sum_{j\le i}P(j)$, the $k$-th draw takes the **smallest** index $i$ with $F_i>u_k$ ("cumulative mass strictly greater"); if floating-point error makes $u_k\ge F_{N-1}$, clamp to $N-1$.
3. **IS weights**: $\tilde w_k=(N\cdot P(\text{indices}[k]))^{-\beta}$, then divide by the batch maximum $\max_j \tilde w_j$.

Return the triple `(indices, weights, probs)`: int64 $(K,)$, float32 $(K,)$, float32 $(N,)$. With an empty buffer ($N=0$) there is no sample to point at, so `indices` is filled with $-1$, `weights` with $0$ and `probs` is empty.

## Examples

`priorities=[1.0, 2.0, 3.0, 2.0]`, `alpha=1.0`, `beta=0.5`, `uniform_draws=[0.05, 0.40, 0.72, 0.95]`:

```
probs   = [0.125, 0.25, 0.375, 0.25]        # prefix sums 0.125 | 0.375 | 0.75 | 1.0
indices = [0, 2, 2, 3]                      # 0.05<0.125→0; 0.40<0.75→2; 0.72<0.75→2; 0.95<1.0→3
weights = [1.00000000, 0.57735026, 0.57735026, 0.70710677]
        # (4·P)^-0.5 = [1.414214, 0.816497, 0.816497, 1.0], divided by max=1.414214
```

Same draws with $\beta=1.0$ (full correction):

```
weights = [1.00000000, 0.33333334, 0.33333334, 0.50000000]
```

With $\alpha=0$ (any $\beta$) $P\equiv 1/N$, all $\tilde w$ are equal, so `weights ≡ 1` and the indices only follow the equal-width bins:

```
probs = [0.25, 0.25, 0.25, 0.25]   indices = [0, 1, 2, 3]   weights = [1, 1, 1, 1]
```

## Constraints

- CPU; $N\le 10^5$, $K\le 10^4$; `priorities` non-negative; $0\le u_k<1$.
- Accumulating internally in float64 and emitting float32 is recommended (checks use `atol=1e-5, rtol=1e-5`; `indices` is compared with `torch.equal`).
- Edges: with $N=0$ (empty buffer, nothing to sample) `indices` is all $-1$, `weights` all $0$ and `probs` is empty; $K=0$ gives empty `indices`/`weights` but `probs` still has length $N$; $N=1$ gives `indices ≡ 0` and `weights ≡ 1`.
- Degenerate invariants: $\alpha=0\Rightarrow$ uniform $P$ and $w\equiv1$; $\beta=0\Rightarrow w\equiv1$; all-equal priorities $\Rightarrow$ uniform $P$ and $w\equiv1$; a zero $p_i$ with $\alpha>0\Rightarrow$ that sample is never picked and $P(i)=0$.
- This problem only handles **sample-level** indices and weights, never token distributions, so it does not overlap the `sampling-*` collection's top-k/nucleus problems; loss and advantage formulas live in the `to-*` collection.
