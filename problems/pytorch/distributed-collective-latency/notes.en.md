# In Depth: Ring vs Recursive-Doubling Latency Accounting

## What the α-β Model Is

The time of one collective splits into two parts: **startup overhead** $\alpha$ (fixed latency per send/recv: software stack, handshakes, kernel launch) and **transmission overhead** $\beta\cdot\text{bytes}$ (bandwidth-limited). Total time $=\text{steps}\cdot\alpha + \text{total bytes}\cdot\beta$. Different algorithms merely trade off "number of steps" against "bytes per step" differently.

## The Trade-off of the Two Topologies

- **Ring**: data is cut into $W$ chunks that travel around the ring. Doing reduce-scatter + all-gather in two passes → $2(W-1)$ steps, each transmitting only $\frac{S}{W}$. Many steps, but **small data per step and full link utilization**, so the bandwidth term $\frac{2(W-1)}{W}S\beta\to 2S\beta$ (almost independent of $W$). Suited to **large messages**.
- **Recursive-Doubling**: at step $i$ each rank exchanges with a partner at distance $2^i$, finishing in $\log_2 W$ steps. Few steps (the latency term $\log_2 W\cdot\alpha$ crushes ring), but each step moves the whole payload. **Requires $W$ to be a power of 2**. Suited to **small messages / high-latency networks** (especially multi-GPU NVLink within a node, where with large $W$ ring's $2(W-1)$ step startups are expensive).

> Note: this problem pins the formulas with a simplified teaching model in which "rd moves $S$ bytes per step"; real RD has a more refined bandwidth term, but the **direction of the trade-off** (latency vs bandwidth) and the conclusion agree. Grading only recognizes the two closed forms given in the statement.

## Intuition for the Crossover

Setting the two equal yields the critical $S^*$: the larger $W$ and the larger $\alpha$, the more rd is favored; the larger $S$ (message), the more ring is favored. This is exactly the basis on which NCCL switches between ring/tree(≈rd) for `all_reduce` based on message size and topology.

## Implementation Pitfalls

- **The tie rule must be pinned down**: $T_{ring}=T_{rd}$ is judged `"ring"` (use `<=`). Otherwise the floating-point equality boundary is ambiguous.
- `speedup = max/min ≥ 1`, always non-negative; do not write fast/slow and end up with <1.
- Use `math.log2` for $\log_2 W$ (exact for powers of 2), avoiding the last-bit difference of `log(x)/log(2)` affecting the tie decision.

## Related

- Upstream: `distributed-all-reduce` / `distributed-reduce` (what is analyzed here is their **elapsed time**).
- Similar accounting: `distributed-pipeline-bubble` (bubble ratio), `distributed-zero-offload` (GPU memory vs bandwidth).
- Grading: pure closed-form algebra + an independent recomputation in `test_cases` (expanded form vs factored form as two paths).
