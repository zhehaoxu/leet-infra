# Solution Notes: Dynamic Sparse Training Drop-Grow Reactivation

## Origin
RigL (Evci et al., *Rigging the Lottery: Making All Tickets Winners*, ICML 2020, arXiv 1911.11134) uses an ERK sparsity distribution for per-layer quotas and, at fixed intervals, "grows by gradient magnitude, drops by weight magnitude"; SET (Mocanu et al., 2018) proposed purely random rewiring earlier. MovERec-style work brings momentum and a **reactivation rate \(\varepsilon\)** into the rule: only a small fraction is replaced each step instead of a full re-ranking, which trains more stably. In this collection `sparsity-cubic-schedule` only covers "how sparsity decays over steps", while this problem covers "how the mask changes in this step" — together they form a complete DST loop.

## Key points and pitfalls
1. **Different sort keys for drop and grow**: drop looks at weight magnitude (smallest leave), grow looks at gradient magnitude (largest enter). Using \(|w|\) for both (or \(|g|\) for both) degenerates into a local reshuffle that can never discover new connections.
2. **When the candidate pool is snapshotted**: reactivation may only draw from elements inactive **in the input mask**. Pruning first and then picking from the current inactive set lets a just-pruned element be pulled straight back, turning the swap into a no-op — the most hidden bug here.
3. **Conservation**: a swap must pair \(r\) drops with \(r\) grows, and \(r\) is capped by both the active count and the pool size so extreme sparsities cannot overshoot.
4. **Definition of \(\varepsilon\)**: the reactivation count is \(\text{round}(\varepsilon\times\text{current inactive count})\), not a fraction of the total; `eps=0` must reactivate nothing at all.
5. **Ties must be deterministic**: equal \(|grad|\) is common (especially after gradient clipping), so break by \(u\) descending and then index ascending to keep judging bit-exact.

## Verification
The reference performs the three phases (top-up / swap) on Python lists and returns sorted indices. The judge's oracle implements the same rule with vectorised `argsort` cascades (its `_rank` applies three stable sorts in reverse priority order — a different mechanism) and compares mask and both index tables bit-exactly. Further assertions: the active count equals \(k\); `eps=0` reactivates nothing and only prunes to \(k\); `sparsity=0/1` yields all-active/all-pruned; reactivated elements were inactive, pruned elements were active, and the sets are disjoint; on equal \(|grad|\) the larger \(u\) wins; output dtype/shape are correct.

## Complexity
\(O(N\log N)\) (two sorts per step); \(O(N)\) memory.
