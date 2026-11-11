# 872 - Implementation Notes

## Boundaries and Exclusions

- Only `phaser` + `_generate_wave_table` are judged; `vad` / `_measure` (SoX cepstral energy gating), `overdrive` / `filtfilt` remain 873+ candidates.
- Orders 871 flanger, 869 biquad, 870 dither, 864 speed/pitch and 716 FSMN-VAD are not re-judged; no real audio IO and no IIR recursion here.

## Numeric and Semantic Details

- The INT wavetable applies `d ± 0.5` then `.to(int32)` truncating toward zero; the FLOAT branch forces binary32, which must be simulated bit-for-bit rather than approximated.
- `delay_pos` and `mod_pos` are independent phases advancing under different moduli; the old `idx` value is read before `delay_pos` is written, and that order cannot be swapped.
- `delay_buf = delay_buf * decay` makes the initial buffer `-0.0` under a negative decay; `_clean` recursively normalises inf/nan but preserves the sign of zero.
- Invalid `wave_type` / `data_type` take the official no-else silent fall-through; `clamp` saturates +/-inf while only nan propagates. 1-D input is accepted by `view(-1,T)` and there is no channel cap.

## Engineering Lessons

- Checks on `int(x+0.5)` need fractional parameters straddling the rounding boundary; an integer-only grid lets tampering survive.
- At `mod_buf_len=2` the SINE table is constant, so different `sample_rate` / `mod_speed` collapse to one output; hardening needs non-constant table sizes and long waveforms.
- torch's vectorised `sin` differs from libm by 1 ulp at n=7/13/14/...; exact `Fraction` derivation of binary32 may also differ by 1 ulp at rounding midpoints, so exact algebraic properties and tolerance-based cross-checks should be asserted separately.
- True-equivalent mutants are not deleted; they are proved by type-sensitive, bit-exact comparisons of the full `solve()` output.
