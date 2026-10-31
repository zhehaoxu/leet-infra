# Solution Notes · AMUS Noisy-Confidence Remasking

## Source

- `huggingface/diffusers`, `src/diffusers/schedulers/scheduling_amused.py` (v0.42.1): `AmusedScheduler.step` (lines 124–191) is the prototype, with `mask_by_random_topk` (lines 31–58), `gumbel_noise` (lines 12–28) and `set_timesteps` (lines 111–122) each supplying one piece. The class docstring describes itself as "A scheduler for masked token generation as used in [`AmusedPipeline`]".
- "Unmask top-k positions step by step" is the shared skeleton of the MaskGIT family (arXiv:2202.04200); the AMUS branch is distinguished by using \(\log p + T\cdot\text{Gumbel}\) as the ranking key, so the hot early steps keep some randomness and nothing gets locked in too soon.
- The library draws randomness twice: the candidate tokens come from `torch.multinomial` (line 149) and the Gumbel noise comes from the `uniform_(0, 1)` inside `gumbel_noise`. Both become explicit inputs here (`drawn_tokens`, `gumbel_uniforms`), removing randomness from the contract and leaving only the deterministic scheduling and comparison.

## Core Points and Pitfalls

**`step_idx` runs opposite to `timestep`.** `set_timesteps` stores `self.timesteps = arange(N).flip(0)` and `step` recovers the index with `(self.timesteps == timestep).nonzero()`, so `step_idx = N - 1 - timestep` and `ratio = (step_idx + 1) / N`. Writing `ratio = (timestep + 1) / N` mirrors the whole schedule: the last step buries the most and the first step barely anything — and since the token block still visibly changes, this is easy to miss in a self-check. `check_schedule_tables` pins the direction by comparing both tables (cosine and linear) step by step.

**The two clamps and what each one means.** After `mask_len = floor(S * mask_ratio)` the source applies `min(sum(unknown) - 1, .)` (comment: "do not mask more than amount previously masked") and then `max(1, .)`. The first guarantees a *net* decrease of at least one mask per step — note the bound is "number of unknown positions minus one", not the count itself. The second is unconditional: even a row with no unknown position at all gets one token buried, and in that case the victim really is an already-committed position. `check_clamped_by_known_count` and `check_no_unknown_still_masks_one` target the two clamps respectively.

**The sentinel is substituted before the logarithm.** The source does `selected_probs = where(unknown, selected_probs, finfo.max)` and only then enters `mask_by_random_topk`, whose first act is `log(clamp(p, 1e-20))`. So a decoded position's confidence is \(\ln(\texttt{finfo.max}) + T g \approx 709.78 + T g\) — it still carries its own noise term. Logging first and then substituting the constant does not flip any result within this task's value ranges, but it shifts the semantics of cases where a decoded position *is* the cut-off, so the contract fixes the source order.

**The cut-off is the element at index `mask_len`, and the test is strict.** `cut_off = sorted_confidence[mask_len]` is the (\(\text{mask\_len}+1\))-th smallest value and the comparison is `<`, not `<=`. With no ties exactly `mask_len` positions are buried; as soon as values tie (three positions sharing the same `gumbel_uniforms` right where the cut-off lands, or a fully decoded row with identical noise) fewer than `mask_len` get buried, possibly none. `check_ties_mask_fewer` and `check_all_tied_nothing_masked` exercise those two degenerate paths.

**Both clamps of the Gumbel transform are needed.** `-log(u)` diverges at \(u = 0\) and `-log(-log(u))` diverges again at \(u = 1\); the two `clamp(1e-20)` calls pin it to \(g \in [-3.829, 46.05]\). The probability side behaves the same: with logits 200 apart the selected probability underflows to 0 and counts as \(10^{-20}\). `check_uniform_clamp` uses \(u = 0 / 0.5 / 1\) together with an extreme logit pair to exercise both clamps.

**`temperature` has two spellings and shares the index with `ratio`.** A tuple/list gives `linspace(t0, t1, N)`; a single number gives `linspace(T, 0.01, N)` (the endpoint is 0.01, not 0). The value used is `temperatures[step_idx]`, not `temperatures[timestep]`. At \(T = 0\) the confidence degrades to plain \(\log p\), which is what makes the hand-checkable "bury the `mask_len` lowest-probability unknown positions" property testable (`check_temperature_zero_picks_lowest_confidence`).

**`timestep == 0` short-circuits the remasking, not the commit.** Candidates still replace the unknown positions and `committed` is still returned; nothing is buried, `masking` is all False and `mask_len` all 0. Returning the original `sample` on the last step loses that commit.

## Deviations from Upstream

- **Precision**: the library runs softmax and the confidences in float32 on CPU, where `torch.finfo(torch.float32).max` logs to 88.72; this contract pins float64, so the sentinel term is 709.78. For \(|T| \le 8\) both give the same masked set (the gap between the sentinel band and any unknown position is far larger than the float difference). Beyond that a sentinel can be outrun, which is why the contract bounds \(|T|\); with the bounds of \(g\), \([-3.829, 46.05]\), the exact safety limit is about 14.2 and 8 leaves headroom.
- **2-D input only**: the `two_dim_input` branch (reshaping and permuting image tokens of shape `(B, S, H, W)` into `(B, H*W, S)`) and `add_noise` (masking by rolling uniform dice against `mask_ratio`) are out of scope.
- **Illegal-input semantics**: the library does not complain when `timestep` is missing from `self.timesteps` — `nonzero()` returns an empty tensor and wrong shapes silently follow; with \(S = 1\) the gather index runs out of bounds and it errors out directly. This task raises `ValueError` throughout and requires \(S \ge 2\) and `timestep` in \([0, N)\) explicitly.
- **Return contents**: with `return_dict=False` the source yields only `(prev_sample, pred_original_sample)`; this task always returns a 4-tuple, exposing `masking` and `mask_len` as observable intermediates, and keeps `mask_len` in the source's `(B, 1)` shape rather than flattening it to `(B,)`.
- The `multinomial` and `gumbel_noise` draws are replaced by explicit inputs; `starting_mask_ratio` stays a parameter (default 1.0) because it enters the `mask_len` formula directly.

## Division of Labor

- `diff-entropy-bound-unmask`: also "which positions commit this step" in discrete diffusion, but there the target is uniform corruption (no MASK token), the criterion is a **prefix sum of temperature-scaled entropies**, rejected positions receive a **uniform random token**, and nothing schedules "how many to bury". This task belongs to the MASK-token family: a **noisy log-probability top-k**, `mask_token_id` for the discarded positions, plus the ratio schedule and its two clamps.
- `diff-discrete-ddim-routes`: the D3PM discrete transition routing (absorbing / uniform corruption matrices) — it computes distributions, not commit sets.
- `diff-repaint-jump-schedule`: decides *when* to redo work in continuous image space via a jump-step table; this task decides *which positions* to redo in discrete token space by confidence.
- `dt-unigram-sample` / `dt-bpe-encode`: discrete choices on the tokenizer side, with no denoising order and no annealed temperature.
