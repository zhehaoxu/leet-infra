# Speculative Decoding: Sequoia Optimal Draft Tree DP with a Depth Bound

## Problem

What a single speculative iteration yields depends on the **shape** of the draft tree. **Sequoia** (Chen et al., 2024, [arXiv:2402.12374](https://arxiv.org/abs/2402.12374)) computes the optimal shape offline by dynamic programming; the sibling problem `spec-sequoia-optimal-tree` (order 177 in this collection) implements its Algorithm 1 — **no depth limit**, acceptance rates indexed by one dimension only ("which child of the parent"), and the tree size fixed in advance.

This problem extends that DP in two directions, matching Sequoia's bounded-depth DP (Appendix C.2.2) and its cost objective (§3.3.1, Eq. (1), speedup $=G(n,d)/(t(n)+d\,c)$):

1. **Depth cap $D$**: a deeper tree than the drafter can actually produce is useless (in self-speculation the drafter only has finitely many layers available), so the tree height is capped at $D$.
2. **Acceptance rates as a 2D matrix indexed by depth**: under self-speculation the acceptance rate of "the $k$-th child" also depends on which level it sits at, so the vector $P[b]$ becomes a matrix $A[d][b]$. Subtrees of the same size are then worth different amounts at different depths, so the DP must carry the **absolute depth** in its state — the translation invariance of Algorithm 1 is gone.
3. **Cost objective**: instead of maximizing the expectation at a fixed size, pick the size $n$ freely in $1\dots N$ and maximize **throughput** $F(n)/(t_{\text{v}}+n\,t_{\text{d}})$, where $t_{\text{v}}$ is the fixed cost of one target verification and $t_{\text{d}}$ the per-node drafting cost.

## Rules

Inputs:
- `A`: acceptance-rate matrix of shape $(D, B)$; $A[d-1][b-1]$ = probability that a node at **absolute depth** $d$ (the root has depth 0) which is the **$b$-th child** of its parent is accepted, conditioned on the whole ancestor chain being accepted.
- `N`: size budget (upper bound on the number of nodes); `B`: maximum number of children per node ($=$ number of columns of $A$); `D`: maximum depth ($=$ number of rows of $A$, may be 0).
- `td`: drafting cost per node; `tv`: fixed cost of one target-model verification ($>0$).

Expected yield of an ordered tree $\mathcal T$ (same accounting as 177):
$$ f(\text{root})=1,\qquad f(v)=\prod_{i=1}^{\text{depth}(v)} A[\,i-1\,][\,\text{child index taken at step }i\,-1\,],\qquad F(\mathcal T)=\sum_{v\in\mathcal T} f(v). $$
Let $F(n)=\max\{F(\mathcal T): |\mathcal T|=n,\ \text{out-degree}\le B,\ \text{depth}\le D\}$ ($-\infty$ if infeasible); the answer is
$$ \max_{1\le n\le N}\ \frac{F(n)}{t_{\text v}+n\,t_{\text d}} . $$

Recurrence of the depth-bounded DP ($d$ is the **absolute depth**):
$$ T[1,0,d]=1,\quad T_{\max}[n,d]=\max_{0\le b\le B}T[n,b,d],\quad T[n,b,D]=-\infty\ (b\ge1), $$
$$ T[n,b,d]=\max_{1\le m\le n-1}\Big(T[n-m,\,b-1,\,d]\;+\;A[d][\,b-1\,]\cdot T_{\max}[m,\,d+1]\Big), $$
i.e. peel off the root's $b$-th child together with its subtree of size $m$ (that child sits at absolute depth $d+1$ and is itself accepted with probability $A[d][b-1]$), while the remaining $b-1$ children stay at depth $d$; the answer is $F(n)=T_{\max}[n,0]$.

Return `(ratio, n_star)`: `ratio` is a float, the optimal objective; `n_star` is the tree size attaining it (int; on ties the smallest $n$).

## Examples

```
A = [[0.75, 0.5],       # acceptance rates of the two branches at depth 1
     [0.5, 0.25]],      # acceptance rates of the two branches at depth 2   => D=2, B=2
N = 4, td = 0.25, tv = 1.0

  F(1) = 1.000  -> 1.000 / (1.0 + 0.25*1) = 0.8
  F(2) = 1.750  -> 1.750 / (1.0 + 0.25*2) = 1.1666666666666667
  F(3) = 2.250  -> 2.250 / (1.0 + 0.25*3) = 1.2857142857142858
  F(4) = 2.625  -> 2.625 / (1.0 + 0.25*4) = 1.3125            <- best
Output = (1.3125, 4)

Same A, free drafting td = 0.0, tv = 1.0 (the depth cap D=2 makes a 4-level chain infeasible, so full size still wins)
Output = (2.625, 4)

A = empty matrix of shape (0, 2) (D = 0, no speculation at all), N = 4, B = 2, td = 0.25, tv = 1.0
Only the root is feasible: F(1) = 1
Output = (0.8, 1)
```

## Constraints

- CPU float32; `A` has shape $(D,B)$ with $0\le A[d][b]\le1$; $1\le N\le 32$, $1\le B\le 8$, $0\le D\le \max(N-1,0)$; $t_{\text d}\ge0$, $t_{\text v}>0$. The return is `(float, int)`; `ratio` is compared with `abs(a-b) <= 1e-6*max(1,|b|)` and `n_star` with integer equality.
- Infeasible sizes ($n>1+B+B^2+\dots+B^D$) stay $-\infty$ in the DP and never win; $n=1$ (root only) is always feasible, so the answer is always defined.
- **Degenerate properties**: with $D=0$ the result is always $\big(1/(t_{\text v}+t_{\text d}),\,1\big)$; with all-zero $A$ every feasible size has expectation $1$, so the optimum is $\big(1/t_{\text v},\,1\big)$; with $B=1$ and all rows equal to $[c]$, $F(n)=\sum_{i=0}^{\min(n-1,D)}c^{i}$.
- Division of labour with `spec-sequoia-optimal-tree` (order 177): when all rows of $A$ are identical, $D\ge N-1$ (the cap does not bind), $t_{\text d}=0$ and $t_{\text v}=1$, this problem's `ratio` must equal 177's $T_{\max}[N]$.
