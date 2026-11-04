# Solution Notes: SCM Trigflow Consistency Step

## Source Localization

- Source file: diffusers `schedulers/scheduling_scm.py` (305 lines, Sana-Sprint team 2025). The core is just the short `step()` body (L274-297): the trigflow `pred_x0` and the "jump back then re-noise" `prev_sample`.
- `set_timesteps` (L130-188) has three construction paths: a custom table (length N+1), the two-step special table `[max_timesteps, intermediate_timesteps=1.3, 0]`, and the default `linspace(max_timesteps, 0, N+1)` (max defaults to 1.57080; a comment notes the sCM paper value is arctan(80/0.5)≈1.56454 and diffusers deliberately chose differently).
- The `step_index` state machine is a copied EulerDiscrete boilerplate (`_init_step_index`/`index_for_timestep`); the comment about "the very first step takes the second index on duplicate values" explains image-to-image start semantics — the judgement domain externalizes this into an explicit argument, keeping only the observable behavior "duplicates allowed".

## Math Notes

1. **Two layers of trigflow**. The angle parameterization `θ = arctan(σ/σ_data)` folds the noise schedule into the unit circle: `x(θ) = cos(θ)·x0 + sin(θ)·σ_data·ε`. The implicit model-output definition is `F(x;θ) = (cos(θ)·x − x0)/sin(θ)`; substituting x gives `F = σ_data·cos(θ)·ε − sin(θ)·x0` — the model predicts both the denoising direction and the data residual. That is why the roundtrip test must synthesize mo as `σd·cos(s)·eps − sin(s)·x0`, not the EDM-style `σd·eps` (the first test draft got exactly this wrong).
2. **Geometry of the consistency step**. `pred_x0 = cos(s)·x − sin(s)·F` is the inner-product inversion pulling the point on the circle back to x0; `prev = cos(t)·x0 + sin(t)·σd·ε'` re-places it at angle t. Multi-step inference is a polyline of jumps on the circle, and the t=0 terminal step is exactly x0.
3. **Role of σ_data**. It appears in the angle definition (arctan denominator), the noise magnitude (randn multiplier), and the training target; at inference it only scales the noise term.
4. **Floating-point exactness**. `torch.cos(0.0) = 1.0` and `torch.sin(0.0) = 0.0` hold exactly, so the terminal step allows a tol=0 bitwise assertion `prev == pred_x0`; `cos(1.57080)` is a tiny ≈7.97e-7, non-zero — the first step leaves a faint noise residue, which is why `cos(s) ≈ 0` must not be simplified to 0 in assertions.

## Implementation Notes

- Validation order: tensor types → shape agreement → timesteps (tensor/1-D/length≥2/float/non-increasing) → step_index (int not bool, range [0, len−2]) → prediction_type → sigma_data (numeric and >0) → noise (required when t≠0, type, shape). Error messages are judgement-domain fixed; reject cases assert message substrings one by one (anti-empty-catch).
- `_as_scalar_angle` accepts 0-dim and single-element tensors (`timesteps[i]` is 0-dim in torch).
- Non-increasing check uses `(diff > 0).any()`, preserving adjacent equals (the official duplicate-value semantics of index_for_timestep).
- When noise=None and t=0, `noise_term` returns an all-zero tensor (`torch.zeros_like`) so the returned dict keeps a stable shape and callers can unconditionally compare it.
- `s_angle/t_angle` are returned as float scalars for easy angle assertions (s from step_index, t from step_index+1 is the most easily swapped detail of this problem — dedicated swap mutations verify it).

## Testing Notes

- Two-step Sana-Sprint default table full-loop comparison (oracle via independent `math.cos/sin`); step 1's oracle input row must take the **actual two column values** of `prev_sample` — do not assume the output keeps the input's `[v, v·0.5−0.3]` construction pattern (the first draft misaligned exactly there).
- `linspace(θ_max, 0, 4)` three-step loop: covers noise terms at intermediate angles.
- One-step `[θ_max, 0]`: noise=None legal, bitwise `prev == pred_x0`, all-zero `noise_term`.
- **noise given but t=0**: two calls with different noises are `torch.equal` bitwise — a direct observation of the multiply-by-zero semantics.
- trigflow roundtrip: synthetic consistency (x0 recovered, tol=1e-12).
- σ_data scaling: `prev(2.0) − prev(1.0) = sin(t)·noise` (per column; the two noise columns of the `_t` construction differ).
- Adjacent-equal table `[0.9, 0.9, 0]`: a legal step with s=t=0.9.
- Purity: two identical calls are bitwise equal (observable consequence of externalizing the official state machine).

## Mutation List (31, all CAUGHT)

Math body 11: cos/sin swaps (on both the sample and mo sides), mo sign flip, s/t swap, t using the current angle, noise coefficient sin→cos, σ_data dropped, prev cos→sin, noise_term sign, zeros_like→pred_x0, returned s_angle→t_angle. Index/bounds 3: upper bound loosened, lower bound loosened, bool slipping through. Validation deletions/loosening 14: monotonic flip (rejecting adjacent equals), monotonic deleted, length deleted, 1-D→2-D, float check bypassed, mo/timesteps type checks deleted, mo shape deleted, noise shape deleted, noise-required deleted, noise-required loosened (rejecting always when t≥0), σ_data boundary loosened, σ_data bool slipping through, prediction_type loosened, scalar numel check inverted. Message 2: step_index range text, noise-required text.

Experience: all 31 CAUGHT with no equivalent mutations — this problem has no randomness and no loop state, a purely deterministic formula, so equivalence risk is naturally low; mutation focus went to structural errors like "s/t indexing" (swap-st and t-cur each caught 8 cases). Lesson from m-tab-tensor SURVIVING the first round: after deleting the timesteps type check, a numpy table explodes with AttributeError (not ValueError), and table_validation had no "timesteps as np" case at the time (the np was passed in the mo position) — **every type check needs its own dedicated reject case; never hitchhike on a neighboring argument's case**.
