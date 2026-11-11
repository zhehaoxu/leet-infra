# 871 - torchaudio Flanger Delay-Line Effect Ledger

This problem reconstructs the flanger effect chain of `torchaudio.functional.filtering`. Two official functions -- `_generate_wave_table` and `flanger` -- are executed verbatim; the judge side reimplements, in pure Python, LFO table generation, the delay-line ring buffer, regen feedback, linear/quadratic interpolation and the final clamp as eight independently judged modes. Everything is deterministic with no RNG and no audio IO; the wavetable is just an explicit formula.

## Source and Source Correction

Order 869 sourced `pytorch/audio main::src/torchaudio/functional/filtering.py` (1702 lines / 30 top-level functions) and executed 11 biquad coefficient functions verbatim; order 870 rescued the `dither` family from the same file. Order 871 takes `flanger` (lines 713-868, 156 lines) and its shared wavetable helper `_generate_wave_table` (lines 16-67, 52 lines).

**Source Correction**: when order 869 was scoped, `flanger`, `phaser` and `vad` were lumped with `dither` into an RNG / wavetable / per-sample-loop exclusion list. In reality `flanger` is fully deterministic with zero RNG: the wavetable is a closed-form `torch.arange` plus `sin` / piecewise-triangle expression, and the loop is a deterministic state machine of length T. Both are exactly reproducible in pure Python.

## Duplicate Check and Scope Boundary

`flanger`, `_generate_wave_table`, `torch.frac`, `regen` and `lfo_length` appear in the corpus only inside the exclusion notes of orders 869 and 870, never as covered mechanisms. Order 864 `au-torchaudio-speed-pitch-ledger` judges resampling and phase-vocoder orchestration; order 716 `asr-fsmn-vad-endpoint` is a FunASR FSMN-VAD endpoint state machine (a different mechanism with a similar name); order 869 judges biquad coefficients and order 870 the 16-bit quantization / noise-shaping ledger. Order 871 re-judges none of them.

## Verbatim Official Environment

`official871.py` extracts the complete source of both functions from the official `.py` via `ast.get_source_segment` and execs them into one shared namespace with `filtering.py::<name>` filenames, WITHOUT changing a single character. It injects `torch` (with `set_default_dtype(torch.float64)`), `math`, `typing`, `Tensor` and `F`. The only instrumentation is a transparent spy on `_generate_wave_table`: it records the exact keyword arguments `flanger` passes and the full returned wavetable tensor, replacing no maths. Census defaults and argument names are read independently from the AST with `literal_eval`.

## Mechanism and the Eight Modes

Five scalings: `feedback_gain=regen/100`, `delay_gain=width/100`, `channel_phase=phase/100`, `delay_min=delay/1000`, `delay_depth=depth/1000`; then two balancing steps `in_gain=1/(1+delay_gain)` and `delay_gain=delay_gain/(1+delay_gain)*(1-|feedback_gain|)`. `delay_buf_length=int((delay_min+delay_depth)*sr+0.5)+2`, `table_min=floor(delay_min*sr+0.5)`, `table_max=delay_buf_length-2`, `lfo_length=int(sr/speed)`.

Wavetable: `phase_offset=int(phase/pi/2*n+0.5)`, `point=(t+phase_offset)%n`; SINE uses `(sin(point/n*2pi)+1)/2`, TRIANGLE uses `point*2/n` followed by the four `floor(4*point/n)` segment cases `+0.5` / `1.5-d` / `1.5-d` / `d-1.5`; then `d*(max-min)+min`. The `FLOAT` branch forces `.to(torch.float32)`, while the `INT` branch keeps float64, applies `d+-0.5`, then truncates to int32.

Per sample: `delay_buf_pos=(pos+L-1)%L` walks backwards, `cur_channel_phase=(channel_idxs*lfo_length*channel_phase+0.5).to(int64)` truncates, and `torch.frac` / `torch.floor` split fractional and integer delay. The buffer is written as `temp+delay_last*feedback_gain`; three taps `delayed_0/1(/2)` feed linear `d0+(d1-d0)*frac` or quadratic `a=d2*0.5-d1`, `b=d1*2-d2*0.5`, `d0+(a*frac+b)*frac` (after `d1-=d0`, `d2-=d0`). The output is `waveform*in_gain+delayed*delay_gain`, then `clamp(-1,1)` and `view(actual_shape)`.

Eight modes: `census` (source census, verbatim defaults and argument names), `wave_table` (per-value table, dtype, phase_offset, points), `flanger` (output, shape, derived quantities and the spy-recorded wavetable call), `branches` (two modulations x two interpolations), `params` (six parameters, three values each), `prefix` (causal prefixes), `errors` (seven exception types and messages) and `shapes` (view packing of 2D/3D/4D inputs).

## Three Quirks Requiring Bit-Exact Replication

(1) **`torch.frac` is sign-preserving**: `torch.frac(x)=x-trunc(x)`, whereas `torch.floor(x)` rounds toward minus infinity. When `delay<0` pushes wavetable values across the negative axis the two disagree, the integer tap is more negative than the fractional part, and interpolation effectively EXTRAPOLATES backwards. Replicating it as `x-floor(x)` fails wholesale on negative-delay configurations.

(2) **float32 wavetable downcast**: even though the main chain runs in float64, the `FLOAT` branch of `_generate_wave_table` still calls `.to(torch.float32)`; the `INT` branch stays float64. The reference must emulate IEEE binary32 rounding bit-for-bit (including signed zero and nan), and must NOT apply that downcast in the `INT` branch.

(3) **Two if/elif chains without else**: an unknown `wave_type` leaves `d` all zeros (the table is constant `min`); an unknown `data_type` leaves `d` in float64 with no downcast. The official code does NOT raise. A reference that raises diverges on out-of-domain behaviour, so both silent fall-through branches must be reproduced explicitly.

## Judging Contract

The judge file is self-contained: 150 official constant snapshots plus 74 `check_*(solve)` functions, 844 lines, using only `math` and `assert`, with no torch / numpy / reference imports. `_deep` compares floats by `repr` plus sign bit, so `-0.0` is distinguishable; non-finite values are normalized to stable strings `"nan"`, `"inf"`, `"-inf"`. The judged domain is 2D/3D/4D waveforms, at most 4 channels, `speed>0` and `lfo_length>0`; all other cases are pinned by the official exception type and message.

## Pipeline and Evidence

Smoke `SMOKE 871 DONE`; 12-stage mechanism probe **1159/1159 ALL GREEN** (independent closed-form derivation, exact Fractions, Newton interpolation identities, deque ring buffer, prefix causality); layer-2 cross-check **146138 checks / 2028 configs / 0 bad**; independent layer-3 oracle **1519 checks / 45 configs / 1429 props / 0 fails** (`cmath.exp` sine, exact Fraction binary32 round-to-nearest-even, exact rational scaling, Newton three-point interpolation).

Judge `driver 74/74 ALL ACCEPTED`; `mut871.py` has 194 mutants, two hardening rounds close 11 judging blind spots (including a long-waveform high-sample-rate `params` case, negative phase, the 4-channel boundary, non-finite cleaning, invalid wavetable types and mode-level snapshots); after excluding 9 real semantic equivalents, **185 active mutants are ALL CAUGHT / 0 SURVIVED / 0 ERROR**. `equiv871.py` proves zero difference over **1370 configs x 9 modes x 9 mutants = 1271025** exact comparisons of the complete `solve()` output (including exceptions, float bits and signed zero).

```text
census / wave_table / flanger / branches / params / prefix / errors / shapes
```
