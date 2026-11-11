# 871 - Implementation Notes

## Scope and Exclusions

- Only `flanger` + `_generate_wave_table` are judged; `phaser` (all-pass chain plus INT wavetable), `vad` / `_measure` (SoX energy gating), `overdrive` and `filtfilt` remain candidates for order 872+.
- Orders 864 speed/pitch, 716 FSMN-VAD, 869 biquad and 870 dither are not re-judged; no real audio IO and no IIR recursion re-judging here.

## Numerical and Semantic Details

- `torch.frac(x)=x-trunc(x)` is sign-preserving while `torch.floor(x)` rounds toward minus infinity. They disagree for negative delay, turning interpolation into extrapolation: the single easiest place to get wrong.
- The wavetable `FLOAT` branch forces float32; the `INT` branch keeps float64 and truncates after `d+-0.5`. Binary32 rounding must be emulated bit-for-bit, not approximated numerically.
- Unknown `wave_type` / `data_type` take the official silent no-else branches instead of raising; `clamp` saturates +-inf to +-1 and only nan propagates, normalized to a string.
- The ring buffer walks backwards with `pos=(pos+L-1)%L` (period L); `lfo_pos` increments by one modulo the table length. Every prefix output must equal the first n columns of the full output bit-for-bit (causality).

## Engineering Lessons

- A short waveform leaves the delay ring unfilled, making different `delay` scan values produce identical output; mode-level checks need a long waveform and a high sample rate to expose a tampered scan table.
- Only a negative phase distinguishes `int()` (truncate toward zero) from `math.floor()`; a non-negative parameter grid misses these branches.
- When a layer-3 oracle derives binary32 from exact Fractions while the official code computes float64 intermediates before the downcast, a 1-ulp difference can appear at a rounding midpoint; assert exact algebra and numeric tolerance separately.
- Real equivalent mutants are never deleted; they are proved by type-sensitive, bit-exact comparisons of the complete `solve()` output.
