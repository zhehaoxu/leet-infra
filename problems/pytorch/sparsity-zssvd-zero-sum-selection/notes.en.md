# Solution Notes

## Approach

The deterministic core of the official mint-vu/Zero-Sum-SVD, from two spots:

1. **planning.py** (L90-207 main selection loop + L383-435 rank/budget constants): whitened spectrum, break-even rank, zero-sum sign-balanced selector, effective_dense_delta accounting;
2. **correction_utils.py** (L97-275 + L289-342): gradient whitening H = G@L⁻ᵀ (triangular solve) and the three δ variants (first/second order, β-mixing).

Out of judging scope: multi-stage planning (inter-stage projection), projected-gradient + re-truncation correction steps, remap quantization, non-zero_sum selection modes; modules always start from full-rank dense.

Boundary with existing problems:
- **773 SVD-LLM**: whitened-domain **per-module fixed-k truncation** (k closed-form from ratio), no global allocation, no gradient;
- **771 LoSparse**: low-rank + structured decomposition, no whitening, no sensitivity, no cross-module budget;
- **772 LLM-Barber**: growth/prune dual thresholds + granularity masks, pruning **weight elements** not singular components.

## Key implementation points

- **Gradient whitening**: H = G@L⁻ᵀ as `solve(L, Gᵀ).t()` — do not invert-then-multiply (numerical stability + matches the official solve semantics).
- **Sensitivity projection**: `grad_σ = diag(Uᵀ H V)` where V comes from the whitened-domain A = W@L. Dropping V (`UᵀH`) or using W's own V are both catchable mutations.
- **Empty preferred heap falls through**: `for prefer in (prefer_pos, not prefer_pos)` — the first draft broke out when the preferred heap was empty, deadlocking/terminating early on sign-imbalanced scenarios, contradicting the official `pop_prefer_then_any` (switch heaps when empty). The extreme construction in `zero_sum_sign_balance` catches this reliably.
- **Three-branch effective_dense_delta**: r>r* recording 0 is not a bug — the low-rank cost (M+N)·r is genuinely more expensive than dense mn above break-even; the mn−r*(M+N) at r==r* settles everything at once. Tests mix shapes (12×8 / 8×16 / 10×10) so all three branches execute.
- **β-mixing δ**: `−sign(g)·|σ|^β·|g|^(1−β)`, with a 0 special case when σ=g=0 (0^0 guard). β=0 and β=1 degenerate to |g|- and |σ|-dominated — tests use β=0.7 to catch exponent swaps (at 0.5 the swap is exactly equivalent, an invalid mutation).
- **Determinism**: candidates iterated over sorted module names, smallest `|δ|` first with name tie-break; the oracle is an independent numpy re-implementation (`np.linalg.svd` + sorting), immune to torch/numpy svd sign conventions (assertions depend only on non-negative σ and δ values).

## Test design (8 cases)

- `zero_sum_oracle`: multiple ratio/keep_ratio/beta/order combos, full-field comparison against the numpy oracle (ranks/masks/truncated/running_sum/dense_removed/selections);
- `delta_order_variants`: second_order half-square term and β-mixing at three levels (0/0.7/1) produce observable selection differences vs first_order;
- `zero_sum_sign_balance`: spectra where one module's δ is always positive and another's always negative; assert running_sum is greedily squeezed near 0 by |δ| plus the empty-heap-fallthrough path (both prefer rounds exercised);
- `break_even_accounting`: per-value assertions on all three accounting branches (constructed so one module trims exactly to r*, another crosses below it);
- `keep_rank_floor`: keep_rank_ratio=1.0 floors rank at r* (not full rank), min_rank upper-bound path, max_svs=0 modules fully excluded from selection;
- `no_whitening_path`: three L=None modules vs the oracle + whitening observability (same weight/grad with and without L yield different ranks);
- `purity`: inputs are not mutated in place;
- `args_validation`: empty dict / missing keys / shape mismatch / non-float / ratio≥1 / keep_ratio>1 / β out of range / invalid order / bools, each with its own message.

## Mutation coverage

41 designed → 39 CAUGHT + 2 equivalence removals:
- **m-cap-neg** (drop `max(0,·)`): min_rank ≤ r* < min(M,N) always holds, so the minuend is always positive and max(0,·) is redundant — the upper-bound semantics is already covered by m-minrank-full;
- **m-val-ratio-bool** (drop the bool guard): True is still rejected by the `< 1.0` branch with identical semantics; False is numerically identical to int 0 and ratio ≥ 0 is legal anyway.

Coverage: whitening dropped/transposed/solve inverted, SVD ascending, projection missing V, δ sign/half-square/order flip/β exponent swap/β sign/β routing, preference flip/always-true preference/no sign heap/tail-takes-head/consumed+2/truncated without square/truncated index, r* formula/min_rank floor/min_rank to full rank, three accounting branches/budget formula/budget freeze/running freeze/dense freeze, seven validation mutations + two message mutations.

## Engineering remarks

- **Selector dead-loop guard**: when the budget is unmet but every module is exhausted, you must break (both prefer rounds empty means no candidates); the oracle and the solution must agree — omitting that break in the oracle caused `cands[0]` IndexError and two false failures.
- `compile()` the mutated source before judging: a replacement string that drops an `if` colon produces a syntax error, and a 0/0 judgement reads as a false SURVIVED (lesson carried from 771).
- σ ≥ 0 is an svd output convention, so `truncated_sigma_sq` simply accumulates σ²; the sign of δ comes only from grad_σ (first order) — that is what makes the sign-heap design work.
