# 859 solution notes: torchaudio sinc-resampling ledger

## 1. Official source

### Files and dependencies

```text
.workbuddy/scratch/captum855/functional.py   2511 lines / 95067 bytes
.workbuddy/scratch/torchaudio859/official859.py   ast x3
WANT = [_get_sinc_resample_kernel,
       _apply_sinc_resample_kernel, resample]
NS injects only: math typing warnings torch Tensor Optional _CPU
```

### Verbatim execution environment

The three functions are `exec`d verbatim into a shared `NS`. The driver layer provides `comp_kernel`, `comp_apply` and `comp_resample`, plus `raw_kernel`, `raw_apply` and `raw_resample`; the latter expose dtype and warning behaviour for the P2 and P10 probes. All tensors default to float64.

## 2. Twelve-segment official probe

### Probe conclusions

P1 pins gcd reduction, `base_freq=min(o,n)*rolloff`, `width=ceil(W*o/base)` and `K=2*width+o`. P2 pins deprecated aliases, invalid methods, non-integer frequencies, `W<=0` and the `<=0` guard of `resample`. P3 pins the identity short circuit returning the original object.
P4 and P5 pin the `-j/n` row offsets, clamp saturation and the `cos(pi/2)**2` residue at the Hann boundary. P6 pins the float32 quantisation of beta, the dual float32/float64 `torch.i0` paths and the `1/i0(beta)` boundary. P7 pins the `t==0` sinc branch and `scale=base/o`.
P8 and P9 check Hann and Kaiser kernels against an independent definitional rebuild (including 160 phases for 44100 to 48000). P10 pins the mutual difference of the float64, `dtype=None` and float32 paths. P11 pins padding, strided convolution, interleave, truncation and 1D/2D/3D unpacking. P12 pins the end-to-end result and the identity short circuit.
Two initial assertions were wrong and were corrected against verbatim measurements: (1) the deprecated alias `sinc_interpolation` does not map to Hann but falls into the Kaiser branch; (2) the `<=0` guard keyword also appears in the builder (from `W<=0`), so the unique string `Original frequency and desired` was used instead. Final result: `PROBE 859 DONE ok=124 fail=0`.

## 3. Three-way verification

### Official layer

The three verbatim functions with torch float64 tensors; `raw_kernel` returns an `(n,1,K)` kernel, and the constant channel axis is squeezed for comparison with the judged `(n,K)` convention.

### Reference layer

Pure-Python `solution_staged.py` with five modes; `f32` reproduces `torch.tensor(float(beta))` via `struct`, `i0_series` uses a power-series recurrence, and `pack` / `unflatten` handle arbitrary leading dimensions.

### Independent oracle layer

An independent definitional rebuild: `math.fsum` for the series terms, explicit per-tap generation, `zip`-based slice dot products and separate shape/flatten/unfold helpers; no reference helper is reused. Since `t`, `window` and `sinc` are ledger-only views not exposed by the official code, they are checked reference versus oracle only.

### Verification result

```text
VERIFY 859 DONE checks=727 fails=0 ALL GREEN
9 parameter cases x (meta + kernel + 4 waveforms x apply/resample)
i0 float64 column: official / reference / oracle
```

## 4. Judge cases and driver

`test_cases_staged.py` holds 35 top-level `check_*(solve)` functions using explicit constants measured in the verbatim official environment: five kernel snapshots (`K84`, `KK84`, `K48`, `K69`, `K84B3`), four output snapshots (`A84_8`, `A84_11`, `A48_8`, `R69`), plus structural assertions for meta, i0, aliases, identity and error messages.
The first judge run had two red cases, both wrong expectations (the reference had always matched the official values): for 8 to 4, `base_freq` is `min(2,1)*rolloff = rolloff` rather than `2*rolloff`; and the `i0_float32` column follows the exact-series-then-quantise convention, while Cephes values from `torch.i0(float32)` had been filled in by mistake. After correction: `DRIVER 859 35/35 ALL ACCEPTED`.

## 5. Mutation testing

`mut859.py` builds 196 mutants covering constants, helpers, every field of `params` / `meta_view`, the formulas and guards of all five modes, and `solve` dispatch. `ANCHOR_INDEX` selects the occurrence for duplicated anchors, and a precheck guarantees zero bad anchors.

```text
round 1: 170 CAUGHT / 25 SURVIVED / 1 ERROR
patch  : +6 hardening checks, 8 equivalence rulings
round 2: 187 CAUGHT / 1 SURVIVED / 0 ERROR
round 3: 188 CAUGHT / 0 SURVIVED / 0 ERROR
196 total / 8 equivalent / 188 active -> ALL MUTANTS CAUGHT
```

The 25 first-round SURVIVED mutants came from untested defaults, error checks that matched only keywords instead of full messages, untested type branches (bool, string, integral float), a sinc ledger column not distinguished from the window column, an unasserted identity short-circuit `meta`, and an unasserted unknown mode. Six hardening groups closed them one by one.
Eight true equivalents: `shape_of` never receives an empty list in the judged domain; `unflatten` with `shape==[]` always sees a one-element slice; the second guard of `require_numbers` still rejects empty arrays with the same message; `is_number` already excludes bool, making the explicit bool check in `require_freq` redundant; the I0 series uses `x*x` so the sign is irrelevant; `pack` only receives data that passed `require_numbers`, so its rank guard is unreachable; and swapping the private `orig`/`new` fields changes no public output (equality is symmetric and `_kernel` / `_apply` re-read the original kw).

## 6. Stripped conventions

(1) The official `(n,1,K)` kernel is stripped to `(n,K)` while `meta.shape` keeps the official 3-D form. (2) The judged domain is fixed to float64, and the `dtype=None` / float32 demotion paths are recorded but not judged. (3) Kaiser beta is limited to the default or to 3.0 / 12.0, which agree bit-for-bit with `torch.i0(float32)`. (4) `i0_float32` is defined as `f32(I0_float64(x))`, not `torch.i0(float32)`. (5) Warning text is simplified to the `deprecated: old -> new` marker. (6) Requiring positive `rolloff` and numeric `beta` are judged-domain constraints, not official guards.

## 7. Official quirk list

Same as section 9 of `description.md`: the deprecated alias landing in the wrong branch, the dual float32/float64 beta paths, the Hann and Kaiser boundary residues, the three dtype paths, the identity short circuit returning the original object, the `<=0` guard living only in the orchestrator, the `ceil(n*L/o)` truncation, and the behaviour contradicting the warning text.

## 8. Deduplication and landing

Before landing, `max_order=858` was confirmed and searches for `sinc_resample`, `sinc_interp_kaiser`, `griffinlim` and `spectral_centroid` returned no hits library-wide. The boundary with 344 is the algorithm family; 855, 857 and 858 share the source but use disjoint function sets. `land859.py` lands 11 files and verifies the three core artifacts against `variants/pytorch/` with `filecmp`.
