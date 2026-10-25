# In Depth: Kernel Launch Timeline and CUDA Graphs

## Source

The PyTorch **CUDA semantics / CUDA Graphs** documentation (`torch.cuda.CUDAGraph`,
`torch.compile(mode="reduce-overhead")`) states that every launch costs the CPU a few microseconds, so a run of
short kernels leaves the GPU "waiting for commands"; capturing the kernels into a graph and replaying it pays that
overhead only once. This problem turns that paragraph into a computable timeline (a deliberately simple
single-machine, single-stream model).

## Key Points and Pitfalls

1. **The readiness time is $(i+1)L$, not $iL$**: the CPU starts issuing at $t=0$, so even kernel 0 must wait for
   its own launch. Using $iL$ magically advances the first kernel by $L$ and breaks both `wall` and the idle count.
2. **Serial recurrence on one stream**: $s_i=\max(r_i,e_{i-1})$, $e_i=s_i+d_i$. The GPU cannot overlap two kernels,
   so a later kernel waits for the previous one; "the CPU finished even earlier" buys nothing extra, it just does
   not block.
3. **busy divides by wall, not by the useful time**: `busy = W / wall`, always $\le 1$; since
   $\text{wall}=W+\text{idle}$, the identity `busy_ratio * wall == sum(durations)` is this problem's invariant probe.
4. **The launch-bound count is an eager-basis diagnostic**: it answers "how many kernels get stuck on a launch on
   the eager timeline", independent of `use_graph` - after packing, no kernel waits for a launch, yet the metric is
   still reported on the eager basis. Whenever $L>0$ it is at least 1 (the very first launch always leaves a gap).
5. **Ceiling of the graph gain**: the graph wall is $L+W$, so $\text{speedup}=\text{eager}/(L+W)$. If every kernel
   dwarfs $L$ then $\text{eager}=L+W$ and the speedup is exactly 1 (a graph cannot rescue a compute/bandwidth
   bottleneck); if all $d_i=0$ the speedup is $n$ (a pure launch bill). Those two degenerate points are hard
   assertions in the grader.
6. **Do not confuse the two clocks**: $d_i$ is GPU execution time, $L$ is CPU launch time - same unit, different
   owner.

## Cross-check Approach

The reference solution runs the forward recurrence for $s_i/e_i$; the grader's `_oracle` uses the classic max-plus
closed form for a single machine with release times,
$e_i=\max_{j\le i}\big[(j+1)L+\sum_{k=j}^{i}d_k\big]$, brute-forced over every start point (triple loop) - a
completely different structure from the recurrence - and re-derives the idle count by comparing $e_{i-1}$ with
$r_i$. Six degenerate assertions cover $n=1$, $L=0$, all-zero durations, all-zero plus zero overhead (both division
guards), long kernels (only the first launch idles, speedup 1), the `busy*wall == sum(d)` identity, and the
invariant that a graph is never slower than eager.

## Complexity

The reference solution scans the $n$ kernels once: $O(n)$ time, $O(1)$ extra space (the float conversion is $O(n)$).
The brute-force oracle is $O(n^3)$ with test $n\le 6$, finishing in well under a second.
