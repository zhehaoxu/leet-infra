# Solution Notes: FasterCache Attention Extrapolation Step

## Source Localization

- Source file: diffusers `hooks/faster_cache.py` (654 lines, FasterCache paper arXiv 2410.19355). This problem takes `FasterCacheBlockHook` (the attention block hook): `new_forward` gate L407-432, `_compute_approximated_attention_output` extrapolation L392-405, cache maintenance L459-478. The denoiser-side CFG-Cache (uncond-branch skip + frequency-domain delta approximation, `FasterCacheDenoiserHook` L261-361 + `_split_low_high_freq` L638) is a separate mechanism with heavy tensor-format branching (BCFHW/BFCHW permute/flatten) and could be its own problem.
- The official gate is doubly negated: `should_compute = iteration > 0 and iteration % N == 0; should_skip = not should_compute` — `>0` is fully subsumed by 0%N==0 (dead code). The judgement domain writes `skip = iteration % N != 0` directly.
- The batch-mismatch recheck: after the initial skip decision, `skip = distilled or state.batch_size != batch` — the cache holds only conditional halves, so a full-batch denoiser forward (uncond+cond) must really run the attention (otherwise shapes mismatch).

## Math Notes

1. **Linear extrapolation**: `t_1 + (t_1 − t_2)·weight` is a first-order approximation of "attention states drifting linearly across timesteps"; weight=0 degenerates to PAB-style direct reuse, weight=1 gives 2·t_1 − t_2.
2. **The interval only permits broadcasting**: `lo < t < hi` strictly on both sides. Endpoint tests must use **conditional-half calls** — a full-batch call inside the range is still bounced to compute by the batch-mismatch recheck, so interval endpoints go untested (the first draft's open_interval used only full-batch calls and three interval mutants SURVIVED).
3. **The divergence condition of two-slot rolling**: new cache = [old[1], new]. Only when old cache[0] ≠ old[1] do `[old[1], …]` and `[old[0], …]` fork — and the first two extrapolations cannot fork because both slots start equal ([c,c] → extrapolation = c); you must reach the fourth layer "compute(A≠B) → skip(extrapolate C) → compute → skip" for the extrapolation reference to be C rather than B. The first draft's full_dance stopped at iter3 and m-cache-slot0 SURVIVED.
4. **Half-batch slice direction**: `[batch:]` takes the latter half (batchwise concat convention: uncond first, cond second); the `[:batch]` mutant needs a dedicated scenario with a full-batch cache plus a conditional-half call.
5. **The cache-write condition**: `not distilled and size(0) == batch_size` before chunking — conditional-half calls (size mismatch) enter the cache as-is; flipping `==` to `!=` puts full batches in unchunked and wrongly chunks halves.

## Implementation Notes

- Validation order: tensor (≥1-D, size(0) is used) → current_timestep → block_skip_range ≥1 → two-integer timestep range → finite weight → distilled bool → state three keys (unknown before missing; cache is "a list of two tensors or None").
- batch_size semantics: recorded as `attention_output.size(0)` on the first call (the true uncond+cond full batch) and kept; the current batch comes from the input every call.
- Functional: new dict + cloned cache entries; the input state untouched.

## Testing Notes

- **The six-layer full dance**: full-batch compute → conditional-half skip (equal slots, extrapolation = c) → full-batch compute (cache=[ca,cb]) → conditional-half skip (cb+(cb−ca)w) → full-batch compute (cache=[approx3, cd], **the slot-rolling fork point**) → conditional-half skip (referencing approx3). Value-by-value assertions.
- Distilled sequence: constant batch, no cache chunking, skip without mismatch.
- weight=0 / weight=1 closed-form endpoints.
- Open-interval endpoints × conditional-half calls (always compute) + mid-range conditional-half (real broadcast).
- A dedicated slice-direction scenario: hand-built full-batch cache + conditional-half call (iteration=1; 0 would compute first).
- Cache slot-link assertions (new[0]==old[1] value-wise), purity, a validation battery (weight=inf/bool, cache length 1/with non-tensors).

## Mutation List (27, all CAUGHT, no equivalents)

Gate 8: open-interval left/right <=, within condition dropped, modulo inverted, modulo shifted, skip recheck dropping distilled/dropping mismatch, batch_size recording broken. Extrapolation 5: negated, reference swapped to t_2, weight dropped, two slice directions. Output/cache 7: skip output swapped to real, compute output swapped to cache, chunk taking the first half, write condition flipped, cache stale, slot link using old[0], iteration not incremented. Validation 7: tensor, dim, skip_range lower bound, range length, weight finite, state missing keys + message.

Experience: ① **when "permission-type" conditions (the interval) stack with "guard-type" conditions (batch mismatch), tests must construct inputs where the permission actually engages** — otherwise all permission-side mutants are shadowed by the guard (the root cause of the first draft's 3 SURVIVED interval mutants); ② the fork point of a two/multi-slot state machine's mutants lies after the state first diverges — sequence cases must walk that far (here calls 4-5); ③ hand-built state's iteration must align with the target branch (iteration=0 hits 0%N==0 → compute first).
