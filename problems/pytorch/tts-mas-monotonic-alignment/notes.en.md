# notes — tts-mas-monotonic-alignment (order 672)

## Source

- Origin: Glow-TTS (2011.06840, the maximum_path Cython implementation in `jaywalnut310/glow-tts` `commons.py`), shared with FastSpeech2 (`ming024/FastSpeech2`); the official version is a parallel DP (diagonal sweep + softmax gating for training). This problem takes the **plain serial DP of its inference hard alignment**: every frame is consumed, text stays or advances by one.
- Boundary with `ctc-loss`: CTC **marginalizes** over all alignments (a sum); MAS extracts a single deterministic alignment by **max**.
- Official MAS requires duration ≥ 1 (each token at least one frame, `T_mel ≥ T_text`); an unreachable endpoint means no solution.

## Judging

- solve args: value [T_mel, T_text]. Returns log_prob / attn / durations.
- Oracle: **exhaustive enumeration of all monotone cuts** (`combinations(range(T_mel-1), T_text-1)` as switch points) computing the max independently, 200 random cases (T_text 1–4, T_mel ≤ 8) cross-checked on log_prob (1e-9) plus durations/attn structural consistency; log_prob is also re-derived from attn as a self-check.
- Anchors: hand-computed 2×2 and 3×1; a greedy≠optimal construction (5/5/100); T_mel == T_text forces the unique diagonal path; a symmetric value (ties: log_prob unique, structure valid); `T_mel < T_text` → ValueError.
- Invalid inputs: empty/ragged/non-numeric/NaN/Inf value → ValueError.

## Mutation log (9/9 caught + 2 documented invalid mutants)

no-continue-branch, no-switch-branch, switch-from-left (switch from the same row's left cell → mel not consumed), negate-value (max becomes min path), parent-flip, duration-off-by-one (first frame not counted), attn-transpose, value-row-swap (mirrored token's value), parent-lookup-flip.

**Documented**: the `if par==1 else` mutant with identical branches (pure syntax equivalence); the endpoint -inf guard is unreachable when T_mel ≥ T_text (equivalent).

## Red lines

No real models/audio (log-likelihood matrix all explicit inputs); the softmax-gated training variant and v2 skip-search are out; cross-checked against `ctc-loss` and the `asr` CTC decoding family.
