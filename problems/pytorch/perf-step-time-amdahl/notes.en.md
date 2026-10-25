# In Depth: Step Time Breakdown, Amdahl and Data Hiding

## Source

Amdahl's law (G. Amdahl, 1967): the machine-wide speed-up is capped by the share you did **not** accelerate,
$\text{speedup}=\big[(1-f)+f/k\big]^{-1}$ with ceiling $1/(1-f)$. The training phase table follows the serial
timeline that the PyTorch Profiler reports (dataloader / fwd / bwd / optimizer). The input-pipeline model mirrors
`torch.utils.data.DataLoader`: `num_workers` are parallel producers and `prefetch_factor` is how many batches each
worker keeps ready - `num_workers=0` is exactly synchronous loading, and a buffer only exists once
`prefetch_factor>=1`. This problem reduces that to a steady-state "supply interval versus compute time" test.

## Key Points and Pitfalls

1. **Amdahl compresses only the named phase**: $\text{new\_step}=(\text{total}-p)+p/k$. Writing `total/k` (as if the
   whole step got faster) is the most common mistake.
2. **`ceiling` is a limit, not a measurement**: $\text{ceiling}=\text{total}/(\text{total}-p)$ corresponds to
   $k\to\infty$, i.e. "erase this phase completely". A 5% phase optimised to infinite speed still buys only 5.3% -
   which is exactly the conclusion of the first example.
3. **$k=1$ must be the identity**: `amdahl` $=1$ and `new_step` $=$ `total`; and `ceiling >= amdahl` always holds.
4. **`compute` excludes the loader**: it is `phase_ms[1]+[2]+[3]`, not `total`. Using `total` as the compute time
   makes the hiding test come out true far too often.
5. **`prefetch` decides whether there is any overlap**: with `prefetch=0` loading is synchronous and you pay the
   full `supply` every step; with `prefetch>=1` you only pay $\max(0,\text{supply}-\text{compute})$. Note that the
   queue *depth* itself does not change the steady-state supply rate - only `num_workers` does, so adding workers is
   the real remedy (consistent with `dataset-dataloader`).
6. **The hiding test includes equality**: $\text{supply}=\text{compute}$ counts as hidden (`data_hidden=True`).
7. **Keep the two views separate**: `fracs` uses the profiler's measured `phase_ms[0]` (the wait that really
   happened), while `effective_step_ms` uses the worker model (what is left once loading is async). That is why
   `phase_ms[0]` never appears in `compute`.
8. **Steady-state amortisation**: `effective` does not charge the warm-up of the first batch; the grader verifies
   that convention with a 4000-step pipeline simulation.

## Cross-check Approach

The reference solution uses the closed forms `total - p + p/k` and `compute = p[1]+p[2]+p[3]`. The grader's
`_oracle` takes different routes: it builds the *modified phase table* (dividing only the target entry by $k$) and
sums it to get `new_step`; accumulates the other entries one by one for the `ceiling` denominator; and derives
`compute` as `total - phase_ms[0]`. Extra assertions cover the $k=1$ identity, $k=10^9$ approaching the ceiling,
`ceiling=inf` plus `amdahl=k` when all other phases are zero, the no-overlap case `prefetch=0`, monotonicity of
`effective` in `num_workers`, `fracs` summing to 1, and a brute-force $S=4000$ step load/compute pipeline
simulation aligned to the closed form.

## Complexity

The reference solution is $O(1)$ (four phases, a handful of divisions). The pipeline-simulation assertion is
$O(S)$ with $S=4000$, costing a few milliseconds.
