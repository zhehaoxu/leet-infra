# 870 - torchaudio Dither Quantization and Noise-Shaping Ledger

This problem reconstructs the dither coefficient-quantization chain of `torchaudio.functional.filtering`. Three official functions -- `_apply_probability_distribution`, `_add_noise_shaping` and `dither` -- are executed verbatim; the judge side reimplements, in pure Python, the 16-bit quantization scale, the three probability-density branches, half-to-even rounding and the first-order error feedback as eight independently judged modes. There is no audio IO and no hidden randomness: every `torch.randint` draw is an EXPLICIT injected input.

## Source and the Source Switch

All 55 top-level functions of `functional.py` (2511 lines, sourced for order 855) were already covered by orders 855-868. Order 869 switched to `pytorch/audio main::src/torchaudio/functional/filtering.py` (1702 lines / 30 top-level functions) and executed 11 coefficient functions verbatim. Order 870 stays in that file and takes the remaining substantive family: the three `dither` functions at lines 465-486 / 489-595 / 598-627.

**Source Correction**: when order 869 was scoped, `dither` was lumped together with `flanger` / `phaser` into an RNG / wavetable / per-sample-loop exclusion list. In reality the TPDF path is fully deterministic (`torch.bartlett_window` plus 16-bit quantization); RNG appears only in the RPDF / GPDF branches and is handled by an RNG spy using the order-865 `torch.rand` injection technique. `_add_noise_shaping` is a purely deterministic first-order recursion.

## Duplicate Check and Scope Boundary

`dither` has only two prior hits in the bank: the Kaldi `_get_window` additive noise step in order 852 `au-kaldi-fbank-compat-ledger` (`noise[i][j]*dither`, a different concept with no quantization, TPDF or noise shaping), and the exclusion note in order 869. `bartlett_window`, `_apply_probability_distribution`, `_add_noise_shaping`, TPDF, RPDF and GPDF have zero hits in all meta, description and notes files.

Order 861 judges the IIR recursion and K-weighting, 869 judges biquad coefficient derivation, 857 judges `add_noise` and other waveform operations, and 852 judges the Kaldi front-end compatibility chain. Order 870 re-judges NONE of these: it judges only the dither quantization scale, the three density branches, half-to-even rounding, the RNG call sequence and noise-shaping recursion.

## Eight Modes

| mode | content |
| --- | --- |
| `quantize` | full quantization chain: out / shape / rng_calls / bartlett / signal_scaled / density / noise_shaping |
| `bartlett` | bit-exact periodic Bartlett triangular window of length n |
| `round` | `torch.round` half-to-even rounding and `x*32766` scaling |
| `noise_shaping` | standalone first-order error feedback and shape ledger |
| `branches` | outputs and RNG call sequences of the TPDF / RPDF / GPDF branches |
| `shapes` | reshape packing / unpacking for 1-D / 2-D / 3-D inputs |
| `errors` | boundaries and exceptions (ragged / empty / unknown density / single-channel GPDF / T=1) |
| `census` | source structure ledger (30 top-level functions, line ranges, four scale constants) |

## Core Mechanics

- Packing: `shape=waveform.size()`, `reshape(-1, shape[-1])` into `(C,T)`; `channel_size=C-1` and `time_size=T-1`.
- Scale: `number_of_bits=16` gives `up_scaling=2**15-2=32766` and `down_scaling=2**15=32768`; `signal_scaled = waveform * 32766`.
- `random_channel` / `random_time` are computed BEFORE the branch (`torch.randint` is called only when `channel_size>0` / `time_size>0`, otherwise 0), yet the TPDF branch never uses them: a classic dead-code quirk.
- TPDF: `torch.bartlett_window(T, dtype=float64)` repeated into `(C,1)` and added to `signal_scaled`. RPDF: a single scalar `waveform[rc][rt]-0.5` broadcast. GPDF: starts from the same cell and accumulates six more `waveform[rand_chan][rand_t]` samples.
- All three branches then share `torch.round` (**half-to-even**, preserving signed zero) -> `/32768` -> `reshape(shape[:-1]+(T,))`. An unknown `density_function` does NOT raise: it falls through to the `else` TPDF branch.
- `dither` calls `_add_noise_shaping` only when `noise_shaping=True`: `error[n]=dithered[n]-waveform[n]`, `error_offset=cat((0, error))` truncated to T, `out[n]=dithered[n]+error[n-1]`; the first sample's error is discarded.

### Two Float / Aliasing Quirks That Must Be Replicated Bit-for-Bit

**Piecewise Bartlett floats**: the periodic window has N=n+1 and factor `f=2/n`; the left half uses `w[i]=i*f` (`i<=n//2`) and the right half `w[i]=2-i*f`. The two halves use different float expressions, so odd n is NOT bit-symmetric: `n=5` gives `[0.0, 0.4, 0.8, 0.7999999999999998, 0.3999999999999999]`; at `n=98` / `196`, `n//2` and `(n-1)//2` differ by 1 ulp. `n=1` is a special case `[1.0]`.

**GPDF in-place view alias**: `gaussian = waveform[rc][rt]` is a 0-D VIEW, so `gaussian += ...` mutates the input waveform. When all six draws hit that cell it doubles each round (`0.1 -> 6.4`). More subtly, `dither(noise_shaping=True)` then passes this ALREADY MUTATED waveform as the original to `_add_noise_shaping`, so the error feedback produces large negative values.

## IEEE Edges and Judging Domain

`inf` / `-inf` / `nan` are not valid Python literals, so fixtures and the reference normalize every non-finite value to a stable string. Ragged waveforms, empty tensors, nested empties, single-channel GPDF and `T=1` error types / messages are pinned verbatim. The judging domain assumes `dithered` and `waveform` share a shape (it is by definition the output of `dither` on the same waveform).

## Pipeline Evidence

| stage | result |
| --- | --- |
| verbatim official env | 3 functions via `ast.get_source_segment`, `torch(float64)` + RNG spy, SMOKE 870 DONE |
| 12 probes | 461 / 461 ALL GREEN |
| layer-2 cross-check | 32035 checks / 822 configs / 0 bad |
| layer-3 independent oracle | 6466 checks / 441 configs / 3050 props / 0 fails |
| judge | 110 official constants / 59 checks / 618 lines / driver 59/59 ALL ACCEPTED |
| mutation | 171 mutants, 11 true equivalents excluded, 160 active all CAUGHT / 0 SURVIVED / 0 ERROR |
| equivalence proof | 1214 configs x 9 modes x 11 mutants = 617507 exact comparisons, zero diffs |

## Rejected

`filtfilt` (two `lfilter` calls plus a flip, too thin), `flanger` / `phaser` / `overdrive` / `_generate_wave_table` (per-sample loops plus wavetables), `gain` / `contrast` / `dcshift` (too thin), `vad` / `_measure` (overlap with `asr-fsmn-vad-endpoint`).
