## Source

- `huggingface/diffusers`, `src/diffusers/schedulers/scheduling_k_dpm_2_ancestral_discrete.py` (750 lines, Apache-2.0). Key lines: the up/down/interpol table construction in `set_timesteps` (L323-336) and the four repeat_interleave expansions (L338-341); the first/second-call branches of `step` (L620-678); `state_in_first_order = self.sample is None` (L528-538); `gamma = 0` (L637-639, `sigma_hat = sigma * (gamma + 1)` is identically sigma).
- Paper: Karras et al. 2022 (KDPM2 sampling as DPM-Solver-2 plus ancestral injection); arXiv 2306.14898 is the standard citation for ancestral diffusion (Ho et al. 2020 DDPM ancestral sampling).
- Same skeleton as 754 (`diff-kdpm2-interleaved-step`, deterministic); this problem only tests the three ancestral-specific mechanism deltas.

## Normalization (judge domain vs upstream)

- **Noise made explicit**: upstream calls `randn_tensor(generator)`; with no pinnable generator semantics, and per repo discipline, it becomes the explicit input `noise` (required on the second call, shape-checked), ignored on the first.
- **State externalized**: upstream keeps `self.sample / self.dt`; here `state` holds only `sample` (upstream's `self.dt` is stored and never read — the second-call branch recomputes `dt = sigma_down - sigma_hat`; dead state, not collected). `state_in_first_order = self.sample is None` maps to `state is None`.
- **Table construction inlined**: upstream builds tables once in `set_timesteps`; here they are recomputed per call (N is small; keeps the function pure).
- Upstream formulas kept verbatim including nan paths: `up[N]`'s 0/0 = nan is not patched (slot never read); `down[N]` and `interpol[N-1]/[N]` are cleared exactly as upstream does.

## Key mechanics

1. **Three deltas** (one-to-one vs 754, directly turned into mutations): midpoint interpolated against sigma_down (vs sigma_next), second-call dt = sigma_down - sigma (vs the full step sigma_next - sigma), and noise * sigma_up injected on landing (vs no noise).
2. **Mathematical redundancy**: down[k] = s_next[k]^2/s[k] and interpol[k] = s_next[k] (exact identities) — upstream's explicit interpol construction is redundant, **but not under float32** (see below).
3. **Nan clearing**: torch's `lerp(finite, -inf, 0.5) = nan` (measured! not -inf and not the convex combination), so both k=N-1 and k=N are nan and upstream's `[-2:] = 0.0` clearing cannot drop a slot; `down[-1] = 0.0` clears the nan at k=N (propagated from up[N] = nan).
4. **Dead assignments on the first call**: upstream fetches `sigmas_up[i]` and `sigmas_down[i-1]` but the first-call branch never reads them; at i=0 `down[-1]` wraps to the tail slot without raising.
5. **No noise on the last logical step**: s_next = 0 implies up = 0; the variance conservation up^2 + down^2 = s_next^2 makes the ancestral step degenerate to a deterministic one there.

## Numerical trap: catastrophic cancellation (a real upstream defect)

In `down[k] = sqrt(s_next[k]^2 - s_up[k]^2)` the two terms are nearly equal when s >> s_next (up approx s_next); the difference is s_next^2 * (s_next/s)^2 — relative error amplified by (s/s_next)^2. Harmless in float64 (this problem); in float32 (the upstream pipeline) a [7.90, 0.205, 0] table measures a **4.2e-5** relative error on down[0] (exact 0.005306328 vs 0.005306550 computed in float32). The stable form is `s_next^2 / s`; the verbatim upstream formula is kept as the spec, with the defect recorded here. Knock-on effect: the interpol midpoint equals s_next in exact math, but under float32 the cancellation error in down makes it **differ** from s_next — so upstream's explicit interpol construction is behaviorally meaningful under float32 and cannot be replaced by s_next (mutation `m-lerp-to-next` is caught).

## Dedup

- **754 `diff-kdpm2-interleaved-step`**: same skeleton, five deltas — interpolation target (down vs next), second-call dt (down - s vs next - s), noise injection (yes vs no), interpol table head (no [0,...] dead head vs yes), number of auxiliary tables (three vs one); the first-call midpoint slot (sigIE[i] vs sigIE[i+1]).
- **522 `diff-euler-ancestral-step`**: single-call Euler ancestral; the noise split into (sigma_down, sigma_up) is isomorphic but has no midpoint and no two-call state machine; the DDPM ancestral whole-loop was already rejected.
- **643 `diff-heun-predictor-corrector`**: derivative averaging vs discarding, arithmetic midpoint vs sigma_down interpolation.
- No overlap with agentic/kvcache.

## Tests and mutations

14 cases: full oracle cross-check (numpy float64 replica of the three-table construction plus the two-call state machine), first-call half-step / stored sample / ignored noise, second-call down-step / noise injection / departure from the stored sample (`prev - noise*up - deriv*dt` recovers the original sample), sigma_down = s_next^2/s and variance conservation, last logical step landing on x0, v_prediction over both calls, four invalid-noise cases, prediction_type validation, parity/range/out-of-bounds (including i exactly 2N and the message assertion "last logical step" to prevent a vacuously-caught range-loosening mutant), four invalid states, six invalid sigma tables, the three-step full loop (2N-1 calls), and an end-to-end run.

33 mutations: **30 CAUGHT + 3 equivalent**:

| Equivalent mutant | Evidence |
|---|---|
| `m-down-tail` (drop `down[-1]=0`) | 300 random tables: interpol elementwise unchanged (the nan is absorbed by `[-2:]`); down[N]=nan lands only in expanded slots 2N-1..2N+1 while the second call reads at most `sig_de[2N-4]` — all unread |
| `m-first-dt-full` (first-call dt from `sig_e[i+1]`) | the deviation of interp[j] from s_{j+1}, relativized through dt, is O(1e-16/(s_next/s)) — at most 1e-15 for any legal input; this is exactly the "interpol == s_next" identity (the 4e-6 float32-domain deviation is absorbed by the relative tolerance) |
| `m-sigma-interpol-guard` (`<= 0` to `< 0`) | for reachable second calls j <= N-2, interp[j] = s_next[j] > 0 (validation forces the first N sigmas positive); min over 300 random tables is 0.016; the zero slots of interpol are only read by first calls, which skip the positivity check |

## Pitfalls

- **Interpol head index**: the ancestral interpol table has no manual `[0,...]` dead head as in 754 — the first call reads `sigIE[i]` (not `i+1`), and first and second calls read the same slot. The first draft copied 754's `i+1` and five cases failed at once; rereading L621 fixed it.
- **Oracle expanded-table index**: the oracle's second call initially used the unexpanded base table's `up[i-1]` — misaligned for j >= 1 (j = 0 coincidentally matched, letting some cases pass vacuously). After expansion `sig_ue[2j] = up[j]`.
- **torch.lerp at -inf**: measured `lerp(finite, -inf, 0.5) = nan` — different from both the numpy formula `a + w(b-a)` (-inf) and the convex combination (-inf). This makes the `[-2:]` clearing a two-slot necessity.
- Guarding reject cases against vacuous catches: after `m-second-range` loosens, i = 2N-1 slides into the "midpoint must be positive" check and still raises ValueError, so the message assertion was changed to "last logical step"; the "first-call index exactly 2N" case guards `m-first-range` (sigma = 0 reaches a division by zero instead of raising).
