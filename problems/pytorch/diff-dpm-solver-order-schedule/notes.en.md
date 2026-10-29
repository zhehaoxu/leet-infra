# Deep Dive: DPMSolverMultistep Order Schedule and Final-Step Downgrade

## Source

HuggingFace diffusers, `src/diffusers/schedulers/scheduling_dpmsolver_multistep.py`: inside `step`, the two booleans `lower_order_final` / `lower_order_second` feed the three-way branch `if self.config.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final`, followed by `self.lower_order_nums += 1` at the end. The formulas come from DPM-Solver++ (Lu et al., 2211.01095); the "downgrade the final steps" heuristic is diffusers' own.

## Key Points and Pitfalls

1. **`N < 15` is a hard threshold**: upstream hard-codes "enable the final-step downgrade when there are fewer than 15 steps" as `lower_order_final and num_inference_steps < 15`. That is why \(N=14\) and \(N=15\) differ entirely in their last two steps (`[2,1]` versus `[3,3]`) — the single easiest thing to gloss over. It is not an open bound and it is not scaled by `solver_order`.
2. **`final_sigmas_type="zero"` triggers the downgrade on its own**: even with `lower_order_final=False` and a large `N`, a zero terminal σ still collapses the last step to first order, because at σ=0 the \(x_0\) conversion degenerates and the high-order differences lose their meaning.
3. **A downgraded final step still advances the counter**: upstream bumps `if self.lower_order_nums < self.config.solver_order: self.lower_order_nums += 1` unconditionally after the branch, so `counters` usually ends at `solver_order` regardless of the order actually used on the last step. In the `N=5, order=3` example, `orders` ends with `[2,1]` while `counters` is still `[..., 3, 3, 3]`. Folding the increment into the "used order 3" branch shifts the whole trace.
4. **`second_final` depends on user flags, not on `floor_final`**: the second downgrade reads the *input* `lower_order_final` and `N < 15`, independent of `euler_at_final` and `final_sigmas_type`. Hence `N=5, lower_order_final=False, final_sigmas_type="sigma_min"` yields `[1,2,3,3,3]`: only the last step is affected, the penultimate one does not drop.
5. **`euler_at_final` only touches the last step**: it feeds the `floor_final` disjunction and nothing else, so with `N=30` it just flips the final entry from 3 to 1.
6. **`"deis"` is a silent alias**: the upstream constructor rewrites `algorithm_type="deis"` to `"dpmsolver++"` instead of erroring. This problem keeps that behaviour, so `solve(5, solver_order=3, algorithm_type="deis")` matches the default configuration item for item.
7. **Noise formulations forbid the zero terminal**: the constructor already rejects `final_sigmas_type="zero"` for `dpmsolver` / `sde-dpmsolver`, since those formulas need an explicit `sigma_min`. So "noise algorithm plus the default zero" must raise `ValueError`, and the tests cover both values.
8. **First order is a floor, not a special case**: with `solver_order == 1` the first branch always wins, yet `counters` still climbs from 0 to 1 and stops, because `low < order` never holds again. `check_counters` sweeps every `steps`/`order` combination and verifies `counters[-1] == min(steps, order)` with per-step increments of only 0 or 1.
9. **Pure integers, no tensors**: the outputs are two python `int` lists, so the float64/float32 precision convention does not apply; what does hold is "`bool` is not an integer", so `solver_order=True` must be rejected.

## Reference Comparison

Judging uses an independent pure-Python oracle (`_oracle`) that re-derives the same branches and compares against the reference solution:

- **Example values**: the `N=5/order=3`, `N=4/default`, and `N=3/order=1` sequences are hard-coded, with an assertion that every element is `int` rather than `bool`;
- **Long trajectory**: `N=20` at full speed yields `[1,2] + [3]*17 + [1]` with counters `[0,1,2] + [3]*18`;
- **Final-step switch**: `lower_order_final` on/off gives `[1,2,3,2,1]` versus `[1,2,3,3,3]` at `N=5`; turning it off at `N=20` still collapses the last step via `final_sigmas_type="zero"` alone;
- **The 15-step boundary**: `N=14` ends `[2,1]`, `N=15` ends `[3,3]`;
- **`euler_at_final`**: at `N=30` only the final entry changes;
- **Counter invariants**: 7 step counts × 3 orders, checking lengths, start, end, per-step increments \(\in \{0,1\}\), and the `<= order` bound;
- **Alias and algorithm families**: `"deis"` matches the default, and both noise algorithms reproduce the same sequence with a `sigma_min` terminal;
- **Combined comparison**: `steps=1..20` × `order=1..3` × 4 flag combinations match the oracle exactly;
- **Invalid inputs**: 13 cases (step count 0/negative/`bool`/float, order 0/4/`bool`, unknown `final_sigmas_type`, unknown `algorithm_type`, noise algorithm with a zero terminal, `lower_order_final=1`, `euler_at_final=None`) must all raise `ValueError`.

Nine assertion groups in total.

## Complexity

\(O(N)\) time and \(O(N)\) space, doing nothing but integer comparisons and increments; no tensor work at all.
