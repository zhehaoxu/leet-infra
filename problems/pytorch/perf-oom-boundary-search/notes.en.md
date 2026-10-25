# In Depth: Turning a Memory Budget into Boundaries - Floors, Quadratic Roots and Divisibility

## Source

- **Peak-memory composition** reuses this bank's conventions: `training-state-memory` / `transformer-mem` supply the
  static footprint (param 2 + grad 2 + master/m/v 12 = 16 bytes per parameter), while the $O(s)$ and $O(s^2)$
  activation terms are what a standard Transformer backward must keep (MLP and residual traffic are linear, the
  attention score matrix is quadratic) - which is exactly why FlashAttention changes the game, by zeroing the
  coefficient of the $s^2$ term.
- **Inverting for micro-batch and sequence length** is the everyday tuning loop of Megatron-LM and DeepSpeed
  (`micro-batch-size`, `max-seq-length`, `global-batch-size = micro * accum * world_size`); here that identity
  becomes an integer test.
- Solving a **quadratic inequality for its largest integer root** is the classic bisection/Newton setting; the
  reference deliberately keeps "closed form plus one correction against the predicate" so that it cross-validates
  with the grader's pure-integer bisection.

## Key Points and Pitfalls

1. **Two kinds of "money" leave the budget**: `reserve_bytes` is subtracted from the **budget**
   (usable = hbm − reserve, never part of the peak) while `cuda_overhead_bytes` belongs to the **peak** (added on
   every `peak` evaluation). Merging them shifts both boundaries by one step.
2. **Feasibility is inclusive**: `peak <= usable` - using memory down to the last byte still counts; the suite has a
   `peak == usable` case, so a `<` would call a perfectly fitting configuration an OOM. `headroom` may be negative
   and is **not clamped** (the negative number is itself the answer to "how much are we short").
3. **The quadratic term also carries m**: the last term is $m\cdot s^2\cdot apq$, not $s^2\cdot apq$ - each sequence
   owns its own score block. Dropping $m$ systematically overestimates `max_micro_batch`.
4. **`max_micro_batch` is a linear floor** - budget divided by the per-micro-batch cost
   $s\cdot ap+s^2\cdot apq$, integer division, no search needed. **`max_seq` is quadratic**:
   $m\,apq\,s^2+m\,ap\,s\le\text{budget}$, whose positive root
   $\frac{-b+\sqrt{b^2+4aR}}{2a}$ must be floored **and then re-checked against the predicate** (float square roots
   can be off by an ulp), otherwise you hand back `bound+1`, a value that overflows the moment you substitute it.
   When $apq=0$ the problem degenerates to a linear floor - do not take a square root there.
5. **Test the sign of the budget first**: if $\text{budget}=\text{usable}-\text{cuda\_overhead}-\text{static}<0$
   even the fixed overhead does not fit, so both boundaries are $0$ (the agreed sentinel for "even 1 does not fit");
   `world_size_needed` is still computed normally, and `global_batch_fits` must be False because `feasible` is.
6. **`world_size_needed` uses the given m, not `max_micro_batch`** - the question is "how many ranks does *this*
   configuration need". Example: with $A=8, m=4$ each rank eats 32 sequences per step; $G=120$ is divisible by
   $8\times15$ (15 being `max_micro_batch`), but not by 32, so 4 ranks are needed and $4\times32=128\ne120$ -
   `global_batch_fits=False`.
7. **Multiply back after the ceiling**: $\text{world}=\lceil G/(A m)\rceil$ only guarantees "enough ranks"; the
   exactness test is $\text{world}\cdot A\cdot m == G$ (equivalently $G\bmod(Am)==0$). Skipping the ceiling and only
   checking divisibility misclassifies $G<A m$ (e.g. $G=31$, $Am=32$).
8. **`act_per_token >= 1` is what keeps the search bounded**: the linear term guarantees a finite upper bound for
   both boundaries, so no extra "infinite" sentinel is needed; relaxing it to 0 would force one.
9. **All-integer comparisons**: every quantity is an int byte count, judged exactly, not with a tolerance. `static`
   is only the sum of the three state terms - do not fold `cuda_overhead` into it (it is peak, not training state).
10. **Division of labour**: the byte counts live in `transformer-mem` / `training-state-memory`, the per-card share
    after sharding in `distributed-zero-shard-ownership`, the bandwidth trade of offloading in
    `distributed-zero-offload`; this problem is only "invert a fixed budget into boundaries".

## Cross-check Approach

The reference solution uses integer floor division for `max_micro_batch`, the quadratic root plus
`while fits(...)` corrections for `max_seq`, the `(G + d - 1)//d` ceiling trick for `world`, and a
**multiply-back** equality test for `global_batch_fits`. The grader's `_oracle` reroutes every step: `static` via
`sum([...])`, the peak with $m$ factored out of the bracket, `headroom` by successive subtraction, **both boundaries
by pure integer bisection** (upper bound `budget // (s*ap) + 1` from the linear term alone, which is always valid,
so no float and no square root anywhere), `world` via "quotient plus remainder carry", and
`global_batch_fits` via a **modulo** test. A third referee lives in the tests: the $\text{peak}(m,s)$ formula is
re-substituted independently to assert that `bound` fits while `bound+1` does not, on top of a one-byte staircase
(budget $k\cdot c$ gives exactly $k$, one byte less gives $k-1$), the $m\leftrightarrow s$ asymmetry, the inclusive
tie, reserve-versus-overhead attribution, negative headroom, and non-divisible $G$.

## Complexity

The reference solution is $O(\log \text{budget})$ in spirit - the ±1 correction after the float root is at most one
step. The grader's integer bisection is $O(\log(\text{budget}/ap))$; the whole suite runs in milliseconds.
