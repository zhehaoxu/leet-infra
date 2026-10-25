# In Depth: Sequence/Context Parallelism Memory-Communication Account

## Source
**Sequence parallelism** from Korthikanti et al. 2022 (arXiv:2205.05198): activations in regions such as LayerNorm/Dropout are split along the **sequence dimension** and coordinated with tensor parallelism, converting between sharded and unsharded regions with All-Gather / Reduce-Scatter. **Context Parallel** for very long contexts (Ring Attention, Liu et al. 2023, arXiv:2304.03278) likewise shards along the sequence dimension and circulates KV around the ranks in a ring. This problem uses a simplified model defined here to quantify "how much memory sharding saves / how much extra communication it costs"; the ring-coefficient convention matches `distributed-zero-communication`.

## Key Points and Pitfalls
1. **Two parts of activation**: only the "shardable" $f\cdot A$ is distributed across the $C$ ranks; the remaining $(1-f)\cdot A$ is replicated in full on every GPU. So per-GPU memory is **not** $A/C$ but $(1-f)A + fA/C$ — when $f<1$ there is a floor of $(1-f)A$ that no number of extra GPUs can shard away.
2. **Communication is two primitives**: one All-Gather to enter the full-sequence view and one Reduce-Scatter to go back, so the factor is $2$, not one. Each rank moves $2 f A\cdot\frac{C-1}{C}$.
3. **Ring coefficient $\frac{C-1}{C}$**: the larger $C$, the closer to 1; "each GPU moves about two copies of the sharded data" is a handy approximation. When $C=1$ the coefficient is 0, degenerating to single-GPU zero communication. Do not write the coefficient as $\frac{C}{C-1}$ or omit the $C=1$ division-by-zero case.
4. **$f=0$ / $f=1$ boundaries**: $f=0$ → fully replicated, no communication; $f=1$ → $A/C$ per GPU, communication $2A\frac{C-1}{C}$. $A=0$ is always 0.
5. **Monotone trade-off**: $C\uparrow$ → per-GPU memory $\downarrow$, communication $\uparrow$, exactly "trading bandwidth for memory".

## Cross-check Approach
The reference solution uses the closed forms $(1-f)A+fA/C$ and $2fA(C-1)/C$; the grader's `_oracle` uses a **step-by-step ring simulation** — splitting the shardable amount $S=fA$ into $S/C$ per step, with reduce-scatter and all-gather each accumulating over $C-1$ steps to give $2(C-1)S/C$; different structure, numerically equivalent (floating point uses the relative tolerance `_close`). Additional assertions cover the $C=1$ degeneration, the $f\in\{0,1\}$ boundaries, and monotonicity with $C$.

## Complexity
\(O(1)\) closed form; the oracle's ring simulation is \(O(C)\).
