# Solution Notes: FasterCache CFG-Cache Frequency Approximation Step

## Source Localization

- Source file: diffusers `hooks/faster_cache.py` (654 lines, FasterCache paper arXiv 2410.19355). Same file as 768 but a separate problem: 768 takes the attention block hook, this one takes `FasterCacheDenoiserHook` L261-361 + `_split_low_high_freq` L638-654 + the weight defaults in `apply_faster_cache` L524-562.
- The official gate: `should_skip_uncond = iteration > 0 and iteration % N != 0 and within` — `>0` is fully subsumed by 0%N==0 (dead code, same pattern as 768); the judgement domain writes the three conditions directly. The official identifiers string pre-check is out of scope.
- The tensor_format branches (BCFHW/BFCHW/BCHW/tuple output) are fixed to BCFHW in the judgement domain; BFCHW is only a permute reordering, mechanically isomorphic.
- The official weight is a callback return value; the default rule in `apply_faster_cache` is "alpha_low/high_frequency applies inside its own `*_weight_update_timestep_range`", which the judgement domain inlines as `_freq_weight`.

## Math Notes

1. **The frequency-domain assumption of CFG-Cache**: the difference `Δ = uncond − cond` between adjacent steps is low-frequency dominated and varies slowly. The full-batch step splits Δ into low/high complex spectra stored in state; the skip step rebuilds `uncond_freq ≈ (Δ_low·w_l + low_cond) + (Δ_high·w_h + high_cond)` — the low-frequency difference is near-constant (slow drift), the high-frequency difference is attenuated/amplified by the alpha weights.
2. **The delta scaling is state-accumulated**: the official `delta <- delta * weight` writes back to state, so k consecutive in-range skips multiply the delta by alpha^k; this is Equation 11's implementation, not a single-step scale. A full-batch step recomputes `delta = low_uncond − low_cond` and overwrites. The ×1.1^4 / ×1.1³ compounding assertions catch "single-step/no scale" mutants.
3. **The circular low-frequency mask**: fftshift moves DC to the center (W//2, H//2) for even sizes; radius `min(H,W)//5` — min because fft2 acts only on the last two dims, H/W being spatial resolution. The mask is inclusive `<= r²`: with r=1 the four points (cx±1, cy), (cx, cy±1) are inside, making the `<` mutant observable.
4. **Round-trip consistency**: ifftshift inverts fftshift; `real(ifft2(full spectrum))` is invertible — with alpha=0 the delta vanishes, uncond_freq == cond_freq, and the rebuilt uncond equals cond elementwise (within 1e-4 float error). A closed-form assertion needing no oracle.
5. **The B*F flattening**: BCFHW → permute(0,2,1,3,4) → (B,F,C,H,W) → flatten(0,1) → (B*F,C,H,W); fft2 acts only on (H,W) with frame and batch dims merged. Restoration unflattens (0,(B,-1)) then permutes back. The numpy-oracle pitfall: with C=1, `reshape(b, *shape[1:])` already yields (B,C,F,H,W) layout, and transposing directly swaps F/C.

## Implementation Notes

- Validation order: tensor → 5-D → integer current_timestep (bool excluded) → skip_range ≥1 (bool excluded) → three ranges (2-tuples of numbers, bool elements excluded) → finite alphas → distilled bool → state four keys (unknown before missing) → skip preconditions (batch_size recorded / half-batch input / delta established) → full-batch consistency (batch matches the record, even).
- The str/bytes pre-check in `_validate_range` is fully subsumed by the list/tuple check (no str is a list/tuple) and was deleted (same redundancy as 766's m-powerexp).
- With `iteration % N` and N=5, full-batch steps only occur at iteration ≡ 0 (mod 5); hand-built state's iteration must align with the target branch.
- Functional: new dict + `detach().clone()` on deltas; the skip step stores the scaled delta (cloned from the product), the full-batch step stores the recomputation.

## Testing Notes

- **The oracle trio** (independent numpy float64): `_np_split` (fft2+fftshift+circular mask), `_np_delta` (full-batch delta), `_np_uncond_approx` (skip rebuild) — complex tensors stacked as real/imag then allclose with rel/atol 1e-3~1e-4 (implementation float32 vs oracle float64).
- The full sequence `[F,T,T,T,T,F]` (N=5) plus a final-iteration assertion.
- Weight ranges: t=400 (low inside, high outside), t=200 (both inside, compounding 1.1^4/1.1³), t=99 / t=301 (open-interval endpoints, ×1.0) — the endpoint cases are the only hunters of m-w-open-lo/hi.
- The alpha=0 closed round trip: delta zeroed → rebuilt uncond == cond.
- Skip preconditions (batch_size unrecorded / delta unestablished / full-batch input), the distilled sequence (delta always None), purity, and a validation battery (5-D, bool timestep, skip_range=0/True, range length/type/bool elements, alpha inf/True, state four-key structure).
- Numerical observability: the deterministic ripple tensors from `_lat` keep the delta elementwise non-trivial; interval/mask mutants produce differences far above the 1e-3 tolerance.

## Mutation List (50: 49 CAUGHT + 1 equivalent excluded)

Gate 5: open-interval left/right <=, modulo inverted/shifted, distilled exemption dropped. Skip preconditions 3: batch_size check deleted, half-batch recheck inverted, delta check deleted. Weights 6: interval left/right <=, inverted, low/high scaling dropped, weights swapped. Frequency split 5: radius //4, center offset, mask strictified, fftshift dropped, high/low masks unified. Skip rebuild 7: ifftshift dropped, `.real` dropped (equivalent, excluded), delta sign flipped, high-frequency term dropped, cat order flipped, state storing the unscaled delta. Flat conversion 2: both permutes dropped. Full-batch branch 8: chunk swapped, delta sign flipped, high frequency using uncond only, batch_size record +1, even check inverted, distilled also computing delta, full-batch output halved, iteration not incremented (skip/full 1 each). Validation 14: tensor, dim, timestep, skip_range lower bound, range length/type/bool elements, alpha finite, negative iteration, batch_size evenness, delta type, state missing/unknown keys, message.

Equivalent excluded (1): `m-real-drop` (removing `.real` from `ifft2(...)`) — the trailing `.to(model_output.dtype)` casting complex to real also exactly discards the imaginary part (`torch.equal(c.to(float32), c.real)` verified True), so `.real` is subsumed.

Experience: ① **the oracle's shape-layout pitfall**: the numpy-side B*F restoration must "reshape(b, f_cnt, -1, …) first, then transpose(0,2,1,3,4)" — transposing on instinct swaps F/C when C=1; ② **subsumed checks in validation guards should be deleted** (the str/bytes check fully covered by the list/tuple check), otherwise the corresponding mutant survives forever; ③ **interval-endpoint mutants need endpoint cases** — the default t=400/200 sit inside the intervals, and only t=99/301 distinguish m-w-open-lo/hi; ④ assert complex-tensor validation messages with key-name substrings (the `state['…']` quote boundary is error-prone).
