# 872 - torchaudio Phaser Modulation Effect Ledger

This problem reconstructs the phaser effect chain of `torchaudio.functional.filtering`. Two official functions -- `_generate_wave_table` and `phaser` -- are executed verbatim; the judge side reimplements, in pure Python, INT wavetable generation, the two-phase ring delay line, decay feedback, two-stage gain and the final clamp as eight independently judged modes. Everything is deterministic with no RNG and no audio IO; the wavetable and the loop are just explicit formulas and a state machine.

## Source and Source Correction

Order 869 sourced `pytorch/audio main::src/torchaudio/functional/filtering.py` (1702 lines / 30 top-level functions) and executed 11 biquad coefficient functions verbatim; order 870 rescued the `dither` family and order 871 rescued `flanger` from the same file. Order 872 takes `phaser` (lines 1206-1296, 91 lines) and its shared wavetable helper `_generate_wave_table` (lines 16-67, 52 lines).

**Source Correction (third in a row)**: when order 869 was scoped, `phaser` was lumped with `flanger`, `vad` and `dither` into an RNG / wavetable / per-sample-loop exclusion list. Order 870 showed the `dither` body is fully deterministic, order 871 showed `flanger` uses zero RNG, and order 872 shows `phaser` is likewise fully deterministic with zero RNG: the INT wavetable is a closed-form `torch.arange` plus `sin` / piecewise-triangle expression and the main loop is a ring-buffer list state machine of length T. Both are exactly reproducible in pure Python, so the loops + wavetable rejection recorded in order 870 is void.

## Duplicate Check and Scope Boundary

`phaser` appears in the corpus only inside exclusion notes and registration ledgers of orders 869 / 870 / 871; `gain_in`, `mod_speed` and `mod_buf_len` have zero coverage. Boundary against order 871: 871 judges a FLOAT wavetable, `frac` / `floor` interpolation, a regen two-level gain balance and a hard 4-channel cap; 872 judges an INT wavetable, integer indexing with no interpolation, single-ring decay feedback, two-stage `gain_in` / `gain_out` and **no channel cap** (five channels work, and 1-D input is accepted by `view(-1, T)`). `phaser` calls neither `biquad` nor `lfilter` nor `allpass_biquad`, so order 869's coefficient family and order 861's IIR recursion are not re-judged; orders 864, 870 and 716 are likewise out of scope.

## Verbatim Official Environment

`official872.py` extracts both complete function sources with `ast.get_source_segment` and `exec`s them into one shared namespace under `torch.set_default_dtype(torch.float64)`; not one character of the official bodies is changed. The only instrumentation is a transparent spy on `_generate_wave_table`: it records the exact kwargs `phaser` passes -- `wave_type=SINE|TRIANGLE`, `data_type=INT`, `table_size=mod_buf_len`, `min=1.0`, `max=float(delay_buf_len)`, `phase=pi/2` -- plus the whole int32 table, replacing no maths. Census defaults and argument names are read independently with AST `literal_eval`.

## Eight Judged Modes

`census` pins source structure and defaults; `wave_table` judges SINE / TRIANGLE / invalid types crossed with INT / FLOAT / invalid types across phases, ranges and sizes; `phaser` judges the full output and derived values; `branches` judges sinusoidal/triangular, decay and gain branches; `params` judges six parameter scans; `prefix` judges causality; `errors` pins seven boundary exceptions; `shapes` judges 2-D / 3-D / 4-D packing.

## Core Mechanics

`waveform.view(-1, T)` packs the batch into `(C,T)`; `delay_buf_len=int(delay_ms*0.001*sr+0.5)` and `mod_buf_len=int(sr/mod_speed+0.5)`; `sinusoidal` selects the wave type. Per sample the official code reads `delay_buf_list[idx]` before writing `delay_buf_list[delay_pos]=temp*decay`, with `idx` and `delay_pos` driven by two independent phases. The output goes through `stack`, `mul_(gain_out)`, `clamp(-1,1)` and `view(actual_shape)`.

```text
phase_offset = int(phase / pi / 2 * n + 0.5)
point       = (t + phase_offset) % n
SINE         d = (sin(point / n * 2 * pi) + 1) / 2
TRIANGLE     d = point * 2 / n ; floor(4 * point / n) -> 4 segments
scaled       = d * (max - min) + min
INT          v = scaled - 0.5 if scaled < 0 else scaled + 0.5 ; int32(v)
idx          = int((delay_pos + mod_buf[mod_pos]) % delay_buf_len)
mod_pos      = (mod_pos + 1) % mod_buf_len
delay_pos    = (delay_pos + 1) % delay_buf_len
temp         = waveform[:, i] * gain_in + delay_buf[idx]
out[:, i]    = clamp(temp * gain_out, -1, 1)
```

## Three Bit-Exact Quirks

1. **INT truncates toward zero**: the `FLOAT` branch of `_generate_wave_table` forces float32, but the `INT` branch stays in float64 and applies `d<0 ? -0.5 : +0.5` before `.to(torch.int32)`, which truncates toward zero rather than flooring; a negative `min` separates the two. 2. **Read before write**: `idx` comes from `delay_pos + mod_buf[mod_pos]` and may equal the write slot; the official code reads the old slot into `temp` before replacing `delay_buf_list[delay_pos]`, so swapping the order is wrong. 3. **Signed zero buffer**: the official code first computes `delay_buf = delay_buf * decay`, so with `decay<0` every initial cell is `-0.0`; with `gain_in=0` the sign of the output zeros is decided by it.

## Numeric and Exception Edges

Invalid `wave_type` / `data_type` take the official no-else silent fall-through without raising (the former leaves d all zeros, the latter keeps float64). `mod_speed=0` gives `ZeroDivisionError: float division by zero`; `sr=0` gives `IndexError: list index out of range`; `delay_ms=0` gives `RuntimeError: ZeroDivisionError`; `delay_ms<0` gives `RuntimeError: zeros: Dimension size must be non-negative.`; `mod_speed<0` gives `RuntimeError: upper bound and lower bound inconsistent with step sign`; ragged input gives `ValueError`. `clamp` saturates +/-inf to +/-1 while only nan propagates and is normalised to a stable string.

## Verification Chain

Twelve mechanism probes `15656 / 15656 ALL GREEN`; layer-2 cross-check `8202 checks / 8202 configs / 0 bad` (type-sensitive, bit-exact floats including `-0.0`); layer-3 independent oracle `63281 checks / 2151 configs / 44836 props / 0 fails` (SINE via `cmath.exp`, TRIANGLE via `Fraction`, binary32 via exact rationals, ring line independently replayed with a dict state machine); **187 official constants / 78 checks / 755-line self-contained judge / driver 78/78 ALL ACCEPTED**; **187 mutants with 183 active ones all CAUGHT after excluding 4 true equivalents** (proved by 1890 configs x 9 modes = 516788 bit-exact full `solve()` output comparisons).

## Engineering Lessons

Parameters shaped like `int(x+0.5)` cannot reveal a tampered `+0.5` on integer inputs, so the judge must pick values straddling the rounding boundary (e.g. `delay_ms=3.45 / 3.499 / 3.7`, `mod_speed=277.0 / 500.0`). Short waveforms or `mod_buf_len=2` (where the SINE table collapses to a constant) let distinct parameters produce one output, so long waveforms and high-sample-rate scans are required. torch's vectorised float64 `sin` differs from libm by 1 ulp at a few `n`: judge constants avoid those sizes and the probe uses 1-ulp-aware assertions. `float64 -> int32` for inf/nan depends on the vectorised tail path, so non-finite ranges are verified only in the FLOAT / BYTE branches. True-equivalent mutants are not deleted; they are proved by type-sensitive, bit-exact comparisons of the full output.
