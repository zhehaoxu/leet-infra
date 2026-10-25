# In Depth: The Batch Curve of Decode Throughput and the KV-Traffic Wall

## Source

- **Large batches do not rescue decode**: Pope et al., *Efficiently Scaling Transformer Inference* (MLSys 2022)
  observes that decoding is memory-bound and its throughput is set by "bytes moved per token": weights are reused
  across the batch, KV caches are not. FlexGen (Sheng et al., ICML 2023) turns the same byte/FLOP accounting into an
  optimal blocking schedule under memory and latency constraints.
- The **per-token KV bytes** $2\cdot L\cdot n_{kv}\cdot d\cdot T\cdot \text{bytes}$ reuse this bank's
  `kv-cache-memory` (as in vLLM/PagedAttention: the cache grows linearly with tokens and is fully re-read each step).
- **$2N$ FLOPs per token** reuses `inference-throughput`; "**arithmetic intensity at or above the ridge counts as
  compute-bound, equality included**" reuses `roofline-arithmetic` (Williams et al., CACM 2009).

## Key Points and Pitfalls

1. **The two terms scale differently with the batch**: the weight read $W=N\cdot\texttt{weight\_bytes}$ is constant
   while the KV read $B\cdot K$ is strictly linear. Since $\text{bytes}/B = W/B + K$ only *decreases* through the
   $W/B$ term, it bottoms out at $K$ - that floor is the wall.
2. **The step time is the max of two candidates**, not their sum: $\max(\text{bytes}/bw,\ B\cdot 2N/peak)$. Any
   piecewise form used instead must be exactly equivalent to that max, or the two sides of the knee will disagree.
3. **The bottleneck test is strict**: `is_memory_bound = flops_step < bytes_step*ridge` with ridge $=peak/bw$. A tie
   counts as compute-bound, matching `roofline-arithmetic`; the suite contains a deliberately tied case
   (`peak_flops == bandwidth`, `context_tokens=0`, `B=1`) so a `<=` is caught.
4. **The ceiling is the min of two caps**: $peak/(2N)$ (compute, batch-independent) and $bw/K$ (the $B\to\infty$ KV
   read cap). For long contexts the second one wins - that is why "7B on an A100 at 2048 context" tops out near
   7450 tokens/s. With $K=0$ only the compute cap exists.
5. **Watch the sign in the knee's closed form**: from $B\cdot 2N\cdot bw\ge peak\cdot(W+B\cdot K)$ you get
   $B\,(2N\,bw-K\,peak)\ge peak\,W$. **If the coefficient is $\le 0$ there is no solution and the answer is -1**,
   meaning no batch ever saturates the compute; dividing through blindly yields a negative or reversed bound.
6. **With `context_tokens=0` the knee is independent of the model size**: $B=peak\cdot\texttt{weight\_bytes}/(2\cdot bw)$,
   i.e. $312e12*2/(2*2e12)=156$ on an A100 in fp16 - the same for 7B and 70B. Counter-intuitive, which makes
   it the best check that the formula was not mis-transcribed.
7. **`balance_batch` is the smallest positive integer solution and a physical quantity**: do not clip it with
   `max_batch` (that parameter only bounds `min_batch_for_slo`). A float-ratio candidate still needs a ±1 correction
   against the predicate itself - the grader's search-based route is exactly the check of that step.
8. **The SLO inversion relies on monotonicity**: $\text{tokens\_per\_s}(B)$ rises in the memory regime and then sits
   exactly at $peak/(2N)$, so it is non-decreasing and binary search is valid. Return -1 both when the target exceeds
   the ceiling and when `max_batch` is too small, and return 1 (never 0 or -1) when `target_tokens_per_s=0`.
9. **`step_time_ms` is milliseconds, throughput is per second** - a factor 1000 apart. `weight_fraction` $=W/(W+BK)$
   is the share of HBM traffic spent on weights; once it collapses, extra batch buys nothing (prefix caching and KV
   quantization exist precisely to shrink $K$).
10. **Division of labour**: single-operator ceilings live in `roofline-arithmetic`, pure-compute throughput in
    `inference-throughput`, KV capacity in `kv-cache-memory`, MoE traffic in
    `distributed-moe-communication-bottleneck`; this problem is only about how the two byte streams move with batch.

## Cross-check Approach

The reference solution takes `step_time = max(bytes/bw, flops/peak)` on totals, gets the knee in closed form as
`ceil(peak*W/(2N*bw-K*peak))` corrected by one step on each side using the predicate, and finds
`min_batch_for_slo` by **binary search**. The grader's `_oracle` re-derives everything else: throughput is computed
per token (divide by the batch first, then `max`, then take the reciprocal), `weight_fraction` becomes
$1/(1+BK/W)$, the ceiling becomes $1/\max(2N/peak, K/bw)$ (which degrades to the compute cap at $K=0$ without an
`inf` branch), the knee comes from **exponential doubling (1,2,4,…) plus bisection** that never touches the closed
form, and `min_batch_for_slo` is a **linear brute-force scan from 1**. Structural assertions cover: throughput
non-decreasing in batch and never above the ceiling, the plateau exactly at the compute cap with step time linear in
batch after the knee, the constant knee 156 and `weight_fraction=1` at zero context, halving of the ceiling (with an
almost unchanged step time) when the context doubles, doubling of the ceiling when GQA heads halve, the SLO knife
edge ($B$ qualifies while $B-1$ does not, -1 when `max_batch` blocks it, -1 above the ceiling, 1 at target 0), and
`balance_batch=-1` with throughput approaching but never crossing $bw/K$ when KV dominates.

## Complexity

The reference solution is $O(\log \texttt{max\_batch})$ for the SLO search plus $O(1)$ knee corrections. The
grader's doubling search is $O(\log B)$ and its scan $O(\texttt{max\_batch})$; the suite keeps `max_batch` in the
low thousands, so everything finishes in tens to hundreds of milliseconds.
