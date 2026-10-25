# Solution Notes · Sequoia Optimal Draft-Tree DP

## Turning "pick a tree shape" into filling a 2-D table

Sequoia's insight: under the **positional acceptance assumption**, the expected number of generated tokens depends only on the tree shape (which child index each node is of its parent) and decomposes into subproblems. The candidate space is therefore described by two integers—the tree size $n$ and the root's **number of direct children** $b$ (FirstBranch)—and we fill the table $T[n,b]$. The answer is $T_{\max}[N]=\max_b T[N,b]$.

## Why the transition looks like this

Consider a tree of size $n$ whose root has $b$ children. Take the root's $b$-th child together with its entire subtree (let its size be $m$); what remains—"root + the first $b-1$ children and their descendants"—is a tree of size $n-m$ with FirstBranch $b-1$, contributing $T[n-m,b-1]$. The detached subtree is actually verified only if **the $b$-th child is accepted** (probability $P[b]$), and its internal optimum is $T_{\max}[m]$, so it contributes $P[b]\cdot T_{\max}[m]$. Maximizing over $m$ gives
$$T[n,b]=\max_{1\le m\le n-1}\big(T[n-m,b-1]+P[b]\,T_{\max}[m]\big).$$
Note that $T_{\max}[m]$ internally uses relative path scores; the child position factor $P[b]$ scales the whole subtree's score proportionally—this is exactly the decomposability of $f(v)=\prod_{\text{Path}} P$.

## Bases and infeasible states

- $T[1,0]=1$: root only—"even without speculation at least 1 token is generated" (bonus / current position).
- $T[n,0]$ is $-\infty$ for all $n>1$: a tree of size greater than 1 cannot have a root with 0 children.
- $b>n-1$ is likewise infeasible (the root's child count cannot exceed the remaining node count) and stays $-\infty$; $T_{\max}$ maximizes only over feasible $b$.

## Degenerating to a chain

When $B=1$ only a chain is possible, $T[N,1]=\sum_{i=0}^{N-1} P[1]^i=\frac{1-P[1]^N}{1-P[1]}$—the classic upper bound on expected acceptance length for single-sequence speculative decoding. With $B\ge2$ branching is allowed and $T_{\max}$ grows roughly logarithmically in $N$ (paper Thm 3.6), which is the core advantage of Sequoia over fixed tree shapes.

## Edge-case self-checks

- `check_example`: concrete values for the chain / branching / root-only configurations.
- `check_chain_formula`: $B=1$ agrees with the closed-form geometric series.
- `check_monotone_in_N`: with all acceptance rates equal to 1, $T_{\max}[N]=N$ (every node is accepted).
- `check_matches_oracle`: small random $N,B,P$, checked against an independent brute-force enumeration (root out-degree + recursive composition of partitions).
