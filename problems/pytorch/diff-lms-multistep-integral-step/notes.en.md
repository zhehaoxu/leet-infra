## Source

- `huggingface/diffusers`, `src/diffusers/schedulers/scheduling_lms_discrete.py` (675 lines, Apache-2.0). Key lines: `step` (L545-623, the three convert branches L587-597, derivative and history pop L600-603, `order = min(self.step_index + 1, order)` L606, the coefficient list L607, the `zip(lms_coeffs, reversed(self.derivatives))` landing L610-612); `get_lms_coefficient` (L267-294, the Lagrange-basis product in `lms_derivative` plus `scipy.integrate.quad(lms_derivative, sigmas[t], sigmas[t+1], epsrel=1e-4)`); `scale_model_input` (L243-265, sample/sqrt(sigma²+1), not part of this problem); the 0 appended at the tail of `set_timesteps` (L344).
- Paper background: linear multistep methods (an Adams-Bashforth family variant over sigma-space); the k-diffusion `sample_lms` (Karras et al. 2022 appendix) is the same construction.
- The `@kohya`-style SpacedSampler in k-diffusion wrappers shares the same structure on VP sigma tables.

## Normalization (judge domain vs upstream)

- **Removing the scipy dependency**: the official `integrate.quad(epsrel=1e-4)`; the judge environment has no scipy, so the solution uses **power-coefficient expansion and analytic integration** (ℓ_c is a polynomial of degree ≤ 3, exact), while the oracle uses **16-node × 8-segment composite Gauss-Legendre numerical integration** — two independent paths cross-validating each other; quad's actual accuracy on smooth polynomials is ~1e-12, far tighter than its epsrel parameter.
- **Externalized state**: the official `self.derivatives` list plus the auto-incrementing `self._step_index`; here `state = {"derivatives": [...]}` with an explicit `step_index`. The history-length contract is made explicit: min(step_index, order) before the call, and `step_index = 0` must pass None.
- **Explicit order**: the official `step` parameter `order` (default 4); an input here, limited to [1, 4].
- Official verbatim formulas kept: all three convert branches (**including `sample` with x0 = mo**, one more than 754/755), the derivative denominator sigma, the reversed `zip` pairing, and the reversed-interval negative-integral quad semantics.

## Key mechanics

1. **The first step degenerates to Euler**: `order_used = min(step_index+1, order)` is always 1 at step 0, where ℓ_0 ≡ 1 and coef[0] = ∫[sigma_0, sigma_1] 1 dtau = sigma_1 − sigma_0 (**negative**), so prev = sample − (sigma_0 − sigma_1)·derivative — coinciding with the deterministic path of 522 `diff-euler-ancestral-step` (but LMS has no noise). With `order = 1` the **entire** run degenerates to Euler (each coefficient is sigma_{t+1} − sigma_t).
2. **coeff[0] pairs with the newest derivative**: `zip(lms_coeffs, reversed(self.derivatives))` — history is appended at the tail and the c=0 coefficient is the Lagrange basis of the newest node. Writing `history[c]` (forward order) happens to agree only while t < order and misaligns as soon as t ≥ order (mutation CAUGHT).
3. **Partition of unity**: Σ_c ℓ_c(tau) ≡ 1 (a basic property of Lagrange bases) → Σ_c coef[c] = ∫[sigma_t, sigma_{t+1}] 1 dtau = sigma_{t+1} − sigma_t (including the final step integrating to 0: sigma_{N−1} − 0). This is a second oracle-independent verification line (asserted directly in a test).
4. **Reversed-interval semantics**: the decreasing sigma table reverses the integration interval [sigma_t, sigma_{t+1}] → individual coefficients may be positive or negative (at order 2, c0 < 0 and c1 > 0), but the **sum is always the interval length** (negative).
5. **No None head in the history**: unlike 756 SA-Solver, the LMS history consists of real derivatives from the first step (the first call already has one), with eviction by popping the head; the contract length min(step_index, order) is a hard check against history misalignment.

## Numerical semantics: replacing quad

The official `quad(epsrel=1e-4)` treats epsrel as a **convergence criterion**, not an accuracy bound — for polynomials of degree ≤ 3 its actual error is ~1e-12. The solution's power-coefficient analytic integration is **exact** (expansion coefficients normalized through the accumulated `denom`), so the two differ at the 1e-12 level, indistinguishable under the 1e-6 float32 output tolerance. The oracle side uses composite Gauss-Legendre (16 nodes × 8 segments) as an independent numerical integration, avoiding sharing the "polynomial expansion" path with the solution.

## Dedup

- **522 `diff-euler-ancestral-step`**: LMS's first step / order=1 coincides exactly with Euler (pinned explicitly by a test here), but LMS's substance is the multistep history plus Lagrange-integral coefficients; Euler's noise split is absent from LMS.
- **754/755 KDPM2**: also "history + coefficients" multistep methods, but KDPM2's coefficients are analytic DPM formulas on an interleaved two-call state machine; LMS is a generic Lagrange integral in a single call.
- **`diff-deis-multistep-coef`**: DEIS coefficients are also integrals (DEIS = the exponential-integral family of Adams-Bashforth), but DEIS precomputes its coefficient table over lambda-space while LMS recomputes from the current sigma nodes on every call.
- **`diff-heun-predictor-corrector`**: single-step derivative averaging, no history.
- **756 SA-Solver**: also has Lagrange polynomials, but SA-Solver is a "Lagrange × exponential integral" sAB/sAM state machine with noise; LMS is a deterministic multistep method with pure polynomial integration.

## Tests and mutations

15 cases: full oracle comparison (an independent numpy replica with composite Gauss-Legendre numerical integration, cross-validating against the solution's analytic integration via two disjoint paths), the first-step Euler degeneration (closed-form coefficient sigma_1 − sigma_0), the order=4 full loop (warmup climb [1,2,3,4,4] + history cap), order=2 cap eviction (the first derivative is squeezed out), the state contract (length exactly min(step_index, order), tampering with the oldest derivative changes the result), order-2 closed forms and partition of unity (c0+c1 = sigma_2 − sigma_1, with the sign assertion c0<0<c1), the final step integrating to 0 (coefficient sum = −sigma_{N−1}), v_prediction / sample full loops, six reject families (prediction_type / eight sigmas / step_index & order / five state cases, all asserting messages), a (1,2) shape, and two examples (a two-step end-to-end plus the order=1 all-Euler run).

33 mutations, **33 CAUGHT + 0 equivalent**: the six sigma checks (length/finiteness/non-negativity/monotone/tail-0/first-N positive), both step_index edges and bool, order range and bool, the prediction_type whitelist, the state length contract / step-0 rejection / shape, all three convert branches (epsilon sign, the v_prediction sample-term sign, the sample branch swapped for sample64), the derivative formula and denominator, the pop-head direction and the cap value, both order_used formulas (+2 and cap −1), the sigma node index, the integration upper bound, the Lagrange skip flip / denominator sign / power-expansion sign, the integration power formula, the landing pairing order (forward vs reversed), the landing launch point (sample vs pred_original), and the landing sign.

## Pitfalls

- **Sign intuition on reversed intervals**: it is tempting to assume "all integral coefficients are negative" — at order 2, c0 = −36/7 while c1 = +8/7 (mixed signs); only the **sum** is constantly the (negative) interval length. The assertion was wrong once; after checking against numerical integration it became the sign assertion c0<0<c1 plus partition of unity.
- **The `_close` shape trap, third strike**: `(1,1)` numpy arrays blow up `float()`, and so do `(1,2)` tensors — just reuse 756's numpy reshape version.
- **The contract decision for an empty state at step 0**: `{"derivatives": []}` at t=0 is mathematically equivalent to None, but the contract rejects it ("state must not be passed at step_index 0") — aligned with 756, avoiding two spellings of the same legal input.
- The `order` passed to `for k in range(order)` upstream is the **post-min order_used** (L606 rewrites the local before L607) — the node count grows with warmup and is not the configured order.
