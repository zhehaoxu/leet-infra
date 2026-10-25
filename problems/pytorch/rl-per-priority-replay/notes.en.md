# In Depth: Prioritized Replay Sampling and IS Weights

## Origin
Schaul, Quan, Antonoglou, Silver, *Prioritized Experience Replay* (arXiv:1511.05952), §3.1: proportional sampling \(P(i)=p_i^\alpha/\sum_j p_j^\alpha\); to correct the induced bias the gradient of sample \(i\) is scaled by \(w_i=(N P(i))^{-\beta}\), normalized by the batch maximum \(\max_j w_j\) (the "divide by \(\max_j (N P(j))^{-\beta}\)" in the paper's footnote / Algorithm 1); \(\beta\) is annealed linearly to 1. Engineering references: OpenAI baselines `segment_tree.py::SampleProb/compute_weight`, rlkit `PrioritizedReplayBuffer.sample_n`.

## Indexing convention (the part implementations get wrong)
Read the sampling probabilities as a **partition of \([0,1)\)**: sample \(i\) owns \([F_{i-1}, F_i)\) with \(F_i=\sum_{j\le i}P(j)\). The hit index is the smallest \(i\) whose prefix sum is **strictly greater** than \(u\), i.e. exactly `np.searchsorted(cdf, u, side="right")` / `torch.searchsorted(cdf, u, right=True)`.
- Using `side="left"` (stop as soon as \(F_i\ge u\)) assigns the **boundary point to the bin that closes at it**: with \(P=[0,0.625,0,0.375]\) the bins are "index 1 owns \([0,0.625)\), index 3 owns \([0.625,1.0)\)", so \(u=0.625\) belongs to index 3 by the half-open convention, while `side="left"` returns 1.
- In floating point \(F_{N-1}\) can come out as `0.9999999999999999`, so a draw near 1 overflows the array; `clamp(max=N-1)` is mandatory.

## Three conventions for the IS weights
1. **Normalize by the batch maximum, not the buffer-wide maximum.** A buffer-wide maximum is never smaller than the batch maximum, so using it shrinks the whole batch by an amount that drifts with buffer state and destroys the verifiable property "the batch's largest weight is exactly 1".
2. **Sign of the exponent**: mainstream code uses \((N P(i))^{-\beta}\) (proportional to \(1/P(i)\) — low-probability samples get high weights, which is what a correction means). The paper's \(\bigl(\tfrac1N\tfrac1{P(i)}\bigr)^{\beta}\) is the same value, but many secondary sources write \((N P(i))^{\beta}\) with the sign flipped. This problem fixes \((NP)^{-\beta}\).
3. With \(\beta=0\) every \((NP)^0\) is 1, so the weights degenerate to all-ones; with \(\alpha=0\) we get \(P\equiv 1/N\), hence \(NP\equiv1\), again all-ones — **two independent routes lead to \(w\equiv1\)**, the cheapest self-check available.

## Degenerate cases and numerical traps
- **All-zero priorities with \(\alpha>0\)**: \(\sum p^\alpha=0\), a raw division yields NaN. The statement pins this down as a fallback to the uniform distribution; implement the explicit branch.
- **\(0^0\)**: both PyTorch and Python's `pow` return 1 when \(\alpha=0\), so even zero priorities degrade to uniform without special-casing — but do not "helpfully" add `p.clamp(min=eps)`, that would break the correct behaviour "a zero priority is never sampled".
- **Accumulation precision**: with \(N=10^5\), a float32 `cumsum` can accumulate \(\sim10^{-3}\) error in the tail bins, enough to misroute a draw sitting near a boundary. Accumulate in float64 and downcast only at the end.
- **Zero-probability bins**: the prefix sum is flat there, so no draw can be strictly inside them — a convenient probe that your hit logic is right.

## Division of labour
- The `to-*` family (`to-gae`, `to-ppo-clipped-surrogate`, `to-grpo-advantage`, …) owns the **formulas**: TD errors, advantages, losses. The `priorities` here are precisely their output (after computing \(|\delta|\)), fed in as input and never re-derived.
- `sampling-*` (top-k / top-p / temperature) operates on the **token distribution over a vocabulary**; this problem operates on **sample-level indices and weights**. They only look alike in "turning a distribution into a discrete choice".
- `rl-async-rollout-staleness` handles *when* samples arrive and *how stale* they are; this one handles *how likely* each sample is to be picked and *what compensating weight* it carries — same data-flow side of RL, orthogonal dimensions.

## Complexity
\(O(N + K\log N)\): one prefix sum plus a binary search per draw. The oracle uses an \(O(NK)\) linear scan with a different accumulation order, so the code paths differ structurally while the results must agree.
