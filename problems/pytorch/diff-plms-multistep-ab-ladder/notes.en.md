## Source

- `huggingface/diffusers`, `src/diffusers/schedulers/scheduling_pndm.py` (504 lines, Apache-2.0). Key lines: `step_plms` (L324-392, ets bookkeeping L365-370, the AB ladder L372-384, `counter += 1` L387); `_get_prev_sample` (L409-458, formula(9): sample_coeff L446, denom L449-451, landing L454-456, `prev_timestep >= 0 else final_alpha_cumprod` L431); the skip_prk_steps branch of `set_timesteps` (L212-215, `concat([:-1], [-2:-1], [-1:])[::-1]` — the second-to-last slot is **duplicated once**); the mode dispatch in `step` (L260-263, `counter < len(prk_timesteps)` goes to prk else plms).
- Paper: Pseudo Numerical Methods for Diffusion Models on Manifolds (Liu et al., arXiv 2202.09778) — the plms AB ladder + formula (9); prk is its 4-substep pseudo Runge-Kutta warmup.
- Stable Diffusion's `PNDMScheduler` defaults to `skip_prk_steps=True` (plms all the way), which is the path this problem pins; the prk branch (12 counters of 4-substep groups with `cur_model_output` accumulation) is out of scope.

## Normalization (judge domain vs upstream)

- **Explicit ᾱ table**: the official `self.alphas_cumprod` (built on the training side) plus `final_alpha_cumprod = alphas_cumprod[0]`; here `alphas_cumprod` is an explicit input and the table head is final_alpha. Validated to lie in (0, 1] and be non-increasing — **ᾱ(0) ≈ 1−β₀ < 1** (0.99907 for the real linear table), not 1: ᾱ(0) = 1 would zero the formula(9) denominator when t = prev.
- **Explicit counter / timestep / step_ratio**: the official `self.counter` auto-increments, the timestep comes from the timesteps table, and δ = num_train_timesteps // num_inference_steps; all three are inputs here. The counter determines the branches (0/1/≥2) and the state contract.
- **Externalized state**: the official `self.ets / self.cur_sample`; here `state = {"ets", "cur_sample"}`. The pre-call ets-length contract [c1→1, c2→1, c3→2, c4→3, c≥5→4] is explicitly validated — the natural shape of the official "truncation inside the call" bookkeeping.
- **prk removed**: the `step_prk` (L265-322) `cur_model_output` accumulation, the `prk_timesteps[counter//4*4]` bookkeeping, and the 12-counter warmup are all out of scope — the criteria match SD usage.
- Official verbatim formulas kept: the four AB coefficient tiers (1/2, 3/2, 23/16/5, 55/59/37/9), the counter-1 (t+δ, t) restart pair, and the double-√ structure of the formula(9) denominator.

## Key mechanics

1. **The counter-1 restart pair**: counter 0 takes a half-real step (t, t−δ) and stashes the sample; counter 1 skips the append, averages the two outputs into a trapezoid, and **re-walks a full step over (t+δ, t) from the stashed sample** — the bookkeeping core of the paper's "the first two plms steps ≡ one trapezoid step". A mutation hotspot (restart / no-append / trapezoid formula, one mutation each).
2. **Pre-call vs post-call ets length**: the `[-3:]` truncation is inside the call — pre-call lengths are [c1→1, c2→1, c3→2, c4→3, c≥5→4] and post-call [1, 1, 2, 3, 4, 4, …]. The first draft wrote the c≥5 pre-call contract as 3 (post-truncation) and was bitten by both the reject case and the full loop — the contract must be accounted from the caller's holdings.
3. **The ladder branches on the post-append length**: four tiers len 1 (counter 0/1), 2, 3, ≥4; counter 2's combination is (3·m2 − m0)/2 (ets = [m0, m2]; m1 is not in the history — counter 1 did not append!).
4. **ᾱ(0) < 1 is formula(9)'s lifeline**: at t == prev the alpha gap is 0 and sample_coeff = 1 → prev ≡ sample (identity degeneration); if ᾱ(0) = 1 and prev lands on the table head, denom = α_t·0 + √(α_t·(1−α_t)·1) is 0 when α_t = 1 — the real table's ᾱ(0) ≈ 0.999 avoids this. On the final step prev_timestep = −δ < 0 → the table head.
5. **The timing of the v_prediction conversion**: ε = √α_t·mo + √(1−α_t)·sample happens inside _get_prev_sample (after the bookkeeping) — at counter 1 the sample is already cur_sample, so the conversion's sample is the restarted one; the mismatch with mo's timestep is upstream behavior kept verbatim.

## Dedup

- **757 `diff-lms-multistep-integral-step`**: both are "history + multistep coefficients", but LMS's coefficients are **integrals of Lagrange bases over sigma-space** (nodes follow the sigma geometry) with a plain head-popping history and no special counter branches; PLMS's coefficients are **fixed rational tiers** (1/2, 3/2, 23/16/5, 55/59/37/9) over alpha-space, with counter 0/1 special cases and a cur_sample restart.
- **756 SA-Solver**: also has the "previous call's stashed sample/noise" triple shift, but SA-Solver is a two-sided sAB/sAM state machine with noise and Lagrange × exponential-integral coefficients; PLMS is one-sided, deterministic, noise-free.
- **`diff-ddpm-step` / `diff-ddim`**: single-step families (no ets history); formula (9)'s landing is in the same domain as DDPM's ancestral posterior mean but PNDM's denom structure differs.
- **`diff-deis-multistep-coef`**: DEIS's coefficients are also AB-family integrals, but over lambda-space and with no counter 0/1 special cases.

## Tests and mutations

13 cases: full oracle comparison (an independent numpy replica), counter 0 identity + stash, counter 1 trapezoid + restart (with a prev_timestep = t assertion), the full-loop AB ladder climb (post-call ets lengths [1,1,2,3,4]), cap eviction (after 7 calls the first output is gone from the history), the final-step table head (t=0 → prev_timestep=−δ → α_prev=ᾱ(0), plus the t=prev identity degeneration prev≡sample pinned by hand), a v_prediction full loop, five reject families (prediction_type / seven ᾱ / ten scalars / twelve state cases including the t+δ overflow, all asserting messages), a (1,2) shape, and two examples (the two-step trapezoid + the AB-coefficient closed forms). State-content independent assertions (the ets tail = this call's raw output, except counter 1 = the previous call's; cur_sample non-None only at counter 0) — guarding the "oracle eats solve's state" blind spot (756's lesson reused directly).

31 mutations, **31 CAUGHT + 0 equivalent**: the four ᾱ checks (length/finiteness/range/monotone — the monotone `>` → `>=` is caught back by the "adjacent-equal is legal" case), both timestep edges and bool, the step_ratio lower bound, counter non-negativity and bool, the prediction_type whitelist, five state checks (c0 rejection, the length contract, c1/c2's cur_sample, shape), three ets-bookkeeping mutations (append condition, truncation width, appended object), seven AB-ladder mutations (trapezoid sign, lost restart, one per 2/3/4-order coefficient, the len2 branch key, the c0 stash), two timestep-bookkeeping mutations (c1's prev, c0's prev), the table-tail overflow guard, and five formula(9) mutations (table head, v_prediction sign, sample_coeff's √, a denom term removed, the landing sign).

## Pitfalls

- **Pre-call vs post-call length**: the first draft's contract said c≥5→3 (booking the truncation pre-call) and was slapped down by both a reject case and the full loop — upstream holds 4 entries and truncates inside the call, so the contract must take the caller's view.
- **ᾱ(0) must not be 1**: the first test draft used a linear interpolation ᾱ(t)=1−0.00098t giving ᾱ(0)=1.0, zeroing the denominator in the final-step hand calculation — switch to the real linear-β cumprod (np.cumprod(1−β)) with ᾱ(0)=1−β₀≈0.9999. The t=prev identity degeneration then becomes a clean closed-form assertion (prev ≡ sample).
- **counter 1's ets tail**: no append → ets[-1] is the **previous** call's output — the state-content independent assertion must fork (counter 1 uses last_mo, the rest use this call's mo).
- **A red baseline's fake CAUGHT**: when first adding the "adjacent-equal is legal" case the timestep ran past the table tail, leaving the baseline at 12/13 and making the mutation list look all-green — fix the baseline first, then rerun everything (753's lesson reused verbatim).
- **The skip table's duplicated slot**: `[-2:-1]` is the **second-to-last** timestep (500 when N=5), not 200 — the first draft duplicated the wrong slot; after recomputing, the table is [999, 500, 500, 250, 0].
