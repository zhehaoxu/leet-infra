# In Depth: Sublinear Activation Checkpointing

## Source
Chen, Xu, Lin 2016, *Training Deep Nets with Sublinear Memory Cost* (arXiv:1604.06174). The paper proves that for an $n$-layer network, $O(\sqrt n)$ checkpoints suffice to reduce activation memory to $O(\sqrt n)$, at the cost of only about one extra forward pass. This problem takes its periodic special case "store a checkpoint every $k$ layers" and turns the optimization of the $\lceil n/k\rceil + k$-type peak function into an integer problem.

## Key Points and Pitfalls
1. **The peak is the sum of two terms**: resident checkpoints $\lceil n/k\rceil$ + the current recompute window $k$. Too small a $k$ → too many checkpoints; too large a $k$ → too wide a recompute window within a segment. Both ends rise, with a valley in between at $k\approx\sqrt n$ and peak $\approx 2\sqrt n$.
2. **Integer ceiling**: use `-(-n//k)` for $\lceil n/k\rceil$, all in integers. `math.ceil(n/k)` goes through floating point and can wobble when $n$ is large (e.g. 40000); the statement requires pure integers.
3. **Ties take the smallest $k$**: scanning in ascending order and updating only on a **strictly smaller** value guarantees this; writing `<=` picks the largest tied $k$ and fails the comparison.
4. **store_all is another baseline**: $n\cdot a$, unrelated to $\text{peak}$'s $\lceil\rceil$ term; for small $n$ (e.g. $n=1$) the checkpoint peak $2a$ is actually **larger** than store-all's $a$ — this is normal. The problem only enumerates $k$ for the minimum of the checkpoint scheme and reports the store-all baseline separately; it does not require one to be smaller than the other.

## Cross-check Approach
The reference solution scans $k$ greedily in ascending order; the grader's `_oracle` uses a **two-pass** structure (first list all peak unit counts, take the minimum with `min`, then trace back the smallest $k$ with `index`), cross-validating the semantics of "ties take the smallest k". Additional assertions pin the boundaries using the analytic property that when $n=m^2$, $best\_k=m$ and $peak=2m$, the linear scaling in $act\_bytes$, and `min_peak<store_all` for deep networks.

## Complexity
Enumeration \(O(n)\); trivial for $n\le 10^5$. It can be reduced to \(O(1)\) with a constant number of candidates near $\lfloor\sqrt n\rfloor$; this problem does not require that.
