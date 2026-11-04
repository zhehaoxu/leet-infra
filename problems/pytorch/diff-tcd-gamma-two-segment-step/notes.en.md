# Solution Notes: TCD Gamma Two-Segment Consistency Step

## Source Localization

- Source file: diffusers `schedulers/scheduling_tcd.py` (810 lines, TCD paper arXiv 2410.05964, 2024-10). The core is `step()` (L594-703).
- `set_timesteps` has two paths: default (original_inference_steps None; k = num_train // original_steps even sampling) and custom timesteps (**strictly decreasing check**: `timesteps[i] >= timesteps[i-1]` raises). The judgement domain tests only the step semantics; the table arrives as an argument.
- The official eta check uses `assert` (AssertionError) with message "gamma must be less than or equal to 1.0" — the judgement domain uses ValueError uniformly but keeps the keyword (reject cases assert the substring).

## Math Notes

1. **The gamma intermediate level is the soul of TCD**. LCM/consistency multi-step sampling does "x_t → x0 → x_prev" each step; TCD changes the re-noising target to the gamma-interpolated level `s = floor((1−eta)·prev_t)`: first `x_s = √ᾱ_s·x0 + √(1−ᾱ_s)·ε` (**the same ε**, preserving the model output's noise direction), then from `x_s` add fresh noise to prev with the DDPM marginal ratio `√(ᾱ_prev/ᾱ_s)`. At eta=0, s=prev_t and the segments merge (deterministic LCM-style); at eta=1, s=0, a full re-noise from x0.
2. **Floor semantics**. `floor(7.5)=7` while round=8 — discriminated by the dedicated eta=0.25, prev_t=10 case (the round mutation is caught there).
3. **Three parameterizations**. epsilon: `x0=(x−√β_t·ε)/√ᾱ_t`; sample: `ε=(x−√ᾱ_t·x0)/√β_t`; v_prediction: `x0=√ᾱ_t·x−√β_t·v`, `ε=√β_t·x+√ᾱ_t·v` (diffusers defines v as `√ᾱ·ε−√β·x0`, i.e. get_velocity — the synthetic cross-check's v formula was first written backwards as `√β·x0−√ᾱ·ε`).
4. **Final-step gate combined with prev_t=0**. The final step (step_index == len−1) takes prev_t=0 and injects no noise; intermediate steps take prev_t from the table. The official `final_alpha_cumprod` (prev<0 branch) is unreachable — overflow yields 0, not −1; dead code, not ported (documented in the docstring).

## Implementation Notes

- Validation order: mo/sample → ᾱ (tensor/1-D/length≥2/float) → timesteps (tensor/1-D/non-empty/integer/**index range first**, strictly decreasing after) → step_index → eta → prediction_type. The index-range check must precede the decreasing check, otherwise `_tab(-1, 0)` trips the decreasing error first (first draft hit exactly this).
- `noise_applied` is exposed as a return value so tests can assert the gate directly (instead of inferring from whether prev changed).
- eta=0 with noise not None: noise is ignored (the official branch never runs, no sampling happens), prev == x_s bitwise.
- Error messages are judgement-domain fixed: "严格递减" for decreasing, "索引范围" for range, the official gamma keyword for eta.

## Testing Notes

- N=5 full-loop comparison (oracle via independent math.sqrt, ᾱ passed as a python list): each step asserts the x0/ε/x_s/prev quadruple plus the timestep_s/prev_timestep/noise_applied metadata.
- **Synthetic cross-check**: synthesize column-wise `x = √ᾱ_t·x0 + √(1−ᾱ_t)·ε` (two different (x0, ε) pairs for the two columns), feed mo per parameterization → x0/x_s must agree to 1e-12. **Never** use `_t`'s linear construction `[v, v·0.5−0.3]` — the mo synthesis formulas (e.g. v's) are non-linear per column, and the linear construction breaks the element-wise relation (first draft failed exactly there).
- Floor discrimination: eta=0.25, prev_t=10 → 7 (round gives 8); eta=0 → identity 10; eta=1 → 0.
- Final step with different noises is bitwise identical (the TCD version of multiply-by-zero: the noise branch is skipped entirely).
- One-step table `[49]`: step 0 is the final step, prev_t=0, s=0.
- eta=0: timestep_s == prev_t, bitwise prev == x_s, noise_applied False.

## Mutation List (34, all CAUGHT, no equivalents)

Gamma interpolation/table location 6: floor→round, eta coefficient halved, s using current t, prev_step_index+2, prev using the current position, overflow taking 1. Parameterizations 5: epsilon sign/denominator swaps, sample numerator coefficient swap, v_prediction sign/coefficient swaps. Two segments 9: x_s coefficient swap, x_s at the prev level, three gate mutations (final gate removed / eta gate loosened / both removed), ratio inverted, noise coefficient duplicated, prev sign flipped, ᾱ_t taken at prev. Validation/messages 14: decreasing flip (allowing adjacent equals), decreasing deleted, range deleted, integer deleted, ᾱ length loosened, step upper bound tightened by 1, eta lower bound deleted, eta message changed, prediction_type tightened, noise-required deleted, noise shape deleted, shape deleted, ᾱ/timesteps type checks deleted, step message changed.

Experience: purely deterministic formulas plus the explicit `noise_applied` return value gave all CAUGHT with no equivalents; structural focuses (s/prev_t indexing, two-segment coefficients, gate direction) were each cross-covered by 2–6 cases. Same lessons as 761: every type check needs its own dedicated reject case (done in one pass this round); synthetic cross-check cases must be constructed column-wise explicitly, never relying on linear pattern construction.
