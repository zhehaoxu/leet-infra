# torchaudio ITU-R BS.1770-4 loudness-gate ledger: K-weighting biquad coefficients, IIR recursion whose clamp applies only to the returned tensor, 0.4 s / 75 percent overlap blocks, absolute and relative dual gating, and five-channel weights

## 1. Background

`torchaudio.functional.loudness` measures audio loudness (LKFS) per ITU-R BS.1770-4: two biquad stages apply K-weighting, then 0.4-second windows with 75 percent overlap produce block energies, an absolute -70 LKFS gate and a relative gate each filter the blocks, and a weighted sum yields the final value. This problem rewrites `loudness` together with its `filtering.py` dependency chain (`treble_biquad`, `highpass_biquad`, `biquad`, `lfilter`) in pure Python and splits it into five independently judged modes.
The strings `loudness`, `LKFS`, `BS.1770`, `biquad`, `lfilter`, `treble_biquad` and `highpass_biquad` have no hits library-wide. The word `loudness` does appear in the notes of order 860, but only in a candidate-avoidance remark claiming partial overlap with the `phase_vocoder` of 857; that remark is inaccurate, since the loudness chain never calls `phase_vocoder`. This problem corrects the record against the official source.

### Official source

```text
pytorch/audio main::src/torchaudio/functional/functional.py
  loudness              (lines ~1535-1593)
pytorch/audio main::src/torchaudio/functional/filtering.py
  treble_biquad / highpass_biquad / biquad / lfilter
  _lfilter / DifferentiableFIR / DifferentiableIIR
  _lfilter_core_generic_loop
```

### Verbatim execution environment

The staging file `official861.py` uses `ast.get_source_segment` to extract `loudness` from the 2511-line `functional.py` downloaded for order 855, and eight functions/classes from the fetched `filtering861.py`, then `exec`s all of them into one shared namespace so `loudness` resolves the real `treble_biquad`, which in turn resolves the real `biquad` and `lfilter`. It injects `math`, `warnings`, `torch`, `torch.nn.functional`, `Tensor` and `Optional`, and sets `_IS_TORCHAUDIO_EXT_AVAILABLE` to False so `_lfilter_core_loop` binds to the generic pure-PyTorch loop. No official body is modified.
`comp_coeffs` temporarily swaps the namespace's `lfilter` for a recorder to capture the official biquad `a` / `b` coefficient triples without changing any body. `loudness_ledger` is a step-by-step diagnostic mirror (not the official function) used to read intermediate ledgers, and `comp_loudness` asserts that its LKFS exactly matches the verbatim `loudness` output.

## 2. Conventions

### Inputs and outputs

```text
solve(mode, **kw) -> dict
mode = coeffs | lfilter | kweight | blocks | loudness
```

The reference exposes only `def solve(mode, **kw)`. The judge file holds top-level `def check_*(solve)` functions into which the judge injects `solve`; `from solution import ...` is forbidden. All inputs are JSON-friendly numbers and nested lists; the implementation is pure Python with no `torch`, `torchaudio`, `numpy` or global RNG.
★ `waveform` for `blocks` and `loudness` is exactly two-dimensional `(channels, time)`; the official `(..., channels, time)` batch axis is outside the judged domain, and the official 0-dimensional LKFS tensor is a Python float in the judged contract. `lfilter` and `kweight` accept any leading dimensions and rebuild the original shape. `blocks` takes an already K-weighted waveform, while `loudness` takes the raw waveform and runs both biquads itself.
Judged numerics are float64 throughout. Official coefficients and `lfilter` follow `waveform.dtype`; the float32 path (coefficients quantized by `torch.as_tensor(..., dtype=float32)`) is recorded but not judged. The independent oracle rewrites the same formulas with `math.pow`, `math.fsum` and explicit padded arrays; all three layers agree with the official implementation to about 1e-15.

### Invalid inputs

Every invalid input raises a `ValueError` with a specific keyword: `must be a non-empty list of numbers`, `must contain only numbers`, `must be a non-empty 2-D list`, `must have at least one column`, `must be rectangular`, `must be a non-empty list`, `must be an int`, `must be >=`, `must be a bool`, `a_coeffs must have exactly 3 numbers`, `b_coeffs must have exactly 3 numbers`, `a_coeffs[0] must be nonzero`, `Only up to 5 channels are supported.`, `waveform must have at least one channel`, `waveform shorter than gate_samples`, `step must be positive (sample_rate too small)` or `unknown mode`. Validation order and exact messages are part of the contract.

## 3. Mode 1: coeffs (K-weighting coefficients)

`treble_biquad` is a SoX-style treble shelf with `gain=4.0` dB, `central_freq=1500.0` Hz and `Q=1/sqrt(2)`; `highpass_biquad` is a second-order highpass with `cutoff_freq=38.0` Hz and `Q=0.5`. Both hand six coefficients to `biquad`, which assembles `a=[a0,a1,a2]` and `b=[b0,b1,b2]` before calling `lfilter`.

```text
A     = exp(gain / 40 * ln(10)) = 10^(gain/40)
w0    = 2*pi*f0/sr            alpha = sin(w0)/(2*Q)
temp1 = 2*sqrt(A)*alpha      temp2 = (A-1)*cos(w0)
temp3 = (A+1)*cos(w0)
treble: b0=A*((A+1)+temp2+temp1)  b1=-2*A*((A-1)+temp3)
        b2=A*((A+1)+temp2-temp1)  a0=(A+1)-temp2+temp1
        a1=2*((A-1)-temp3)        a2=(A+1)-temp2-temp1
highpass: b0=b2=(1+cos(w0))/2  b1=-1-cos(w0)
          a0=1+alpha  a1=-2*cos(w0)  a2=1-alpha
```

Each of `treble` and `highpass` returns raw `a` / `b`, `a_norm` / `b_norm` divided by `a0`, plus `w0` / `alpha`; treble additionally exposes `A` / `temp1` / `temp2` / `temp3`. The highpass satisfies `b0==b2` and `a0+a2==2`. `sample_rate` must be an int >= 1 (bool is rejected). The official code does not check Nyquist, so at sr=25 the 1500 Hz shelf is far above Nyquist and the coefficients remain numeric but the filter is unstable; the judged waveform domain therefore uses sr>=4000.

## 4. Mode 2: lfilter (IIR difference equation)

Official `lfilter` first sends `b/a0` to `DifferentiableFIR` (an `F.conv1d` cross-correlation equivalent to a left-zero-padded FIR), then sends `a/a0` to `DifferentiableIIR` (a per-sample feedback loop). Initial conditions are zero, and the recursion is:

```text
z[n] = bn0*x[n] + bn1*x[n-1] + bn2*x[n-2]
y[n] = z[n] - an1*y[n-1] - an2*y[n-2]   (x[-1]=x[-2]=0, y[-1]=y[-2]=0)
out[n] = clamp(y[n], -1, 1) if clamp else y[n]
```

★ Key quirk: `clamp` is applied by `lfilter` to the whole output tensor **after** `_lfilter` returns, so the recursion keeps the **unclamped** `y[n]`. Feeding the clamped value back therefore changes later samples and is a common wrong answer. `lfilter` also packs any leading dimensions into `(batch, channels, time)` and reshapes back afterwards.

## 5. Mode 3: kweight (two biquad stages)

`loudness` runs `treble_biquad(waveform, sr, 4.0, 1500.0, 1/sqrt(2))` then `highpass_biquad(treble, sr, 38.0, 0.5)`, both through `lfilter(clamp=True)`. The mode returns `after_treble` and `after_highpass` (the K-weighted waveform) plus the coefficients. Because clamp bounds to [-1, 1], waveforms with amplitude above 1 are clipped before entering the next stage.

## 6. Mode 4: blocks (blocking and dual gating)

```text
gate_samples = int(round(0.4 * sr))   step = int(round(gate_samples * 0.25))
n_blocks     = (T - gate_samples)//step + 1
energy[c][b] = mean(x[c][b*step : b*step+gate_samples]^2)
g            = [1.0, 1.0, 1.0, 1.41, 1.41][:channels]
ew[b]        = sum_c g[c]*energy[c][b]
block_loudness[b] = -0.691 + 10*log10(ew[b])
gate_abs[b]  = block_loudness[b] > -70.0
efa[c]       = mean over gate_abs blocks of energy[c][b]
ew_abs       = sum_c g[c]*efa[c]
gamma_rel    = -0.691 + 10*log10(ew_abs) - 10
gate_rel[b]  = gate_abs[b] and block_loudness[b] > gamma_rel
efr[c]       = mean over gate_rel blocks of energy[c][b]
LKFS         = -0.691 + 10*log10(sum_c g[c]*efr[c])
```

★ Channels 4 and 5 carry the LFE weight 1.41, and at most 5 channels are supported; more raises `Only up to 5 channels are supported.`. `unfold` keeps only complete windows, so a trailing partial block is dropped. The absolute gate is a **strict** greater-than against -70; the relative gate sits 10 dB below the mean loudness of the absolutely gated blocks. If a stage selects no block, `0/0` yields nan, `log10(nan)` stays nan, and the whole chain ends in nan rather than raising.
`step` uses Python `round` (banker's rounding): at sr=25, `gate_samples=10` gives `round(2.5)=2`; at sr=35, `gate_samples=14` gives `round(3.5)=4`, whereas truncation would give 2 and 3. When every block survives both gates, `LKFS == gamma_rel + 10` holds identically.

## 7. Mode 5: loudness (full chain)

`loudness` = `kweight` followed by the full gating ledger of `blocks`, returning additionally `after_treble`, `filtered`, `coeffs` and `shape`. The verbatim official `loudness` returns a 0-dimensional tensor for 2-D input; the judged contract strips it to a float. Example (sr=4000, 1 channel, 2000-sample square wave): `gate_samples=1600`, `step=400`, `n_blocks=2`, block loudness about `[-6.794206, -6.783260]`, and final `LKFS=-6.788729656623073`.

## 8. Official quirk list

(1) The `waveform.size(-2) > 5` check reads the channel axis, so a 1-D waveform raises `IndexError` rather than `ValueError`. (2) `treble_biquad` writes `sin(w0)/2/Q` while `highpass_biquad` writes `sin(w0)/2.0/Q`; different spelling, same value. (3) `A = exp(gain/40*ln(10))` rather than a literal `10^(gain/40)`. (4) The `lfilter` clamp does not participate in the recursion. (5) Each of the two biquads clamps once. (6) The `g` tensor follows `waveform.dtype`. (7) `count_nonzero(gated_blocks, dim=-1)` on `(...,1,B)` counts only blocks. (8) The absolute gate is strictly greater than -70. (9) `gamma_rel` uses the energy of **absolutely gated** blocks, not all blocks. (10) The relative gate is a `logical_and` rather than a fresh threshold. (11) A silent block gives `log10(0)=-inf`, and when all blocks are dropped the whole chain becomes nan. (12) `gate_duration=0.4`, `overlap=0.75`, `gamma_abs=-70.0` and `kweight_bias=-0.691` are literal constants. (13) `sample_rate` is not checked against Nyquist.

## 9. Example and landing

```text
solve('coeffs', sample_rate=8000)['highpass']['a']
  -> [1.0298406997386809, -1.9991093343177666, 0.9701593002613191]
solve('blocks', waveform=[[0.0]*10 + [1.0]*10,
                         [0.0]*10 + [0.5]*10], sample_rate=25)
  -> n_blocks=6, gate_abs=[False,True,True,True,True,True]
     LKFS=-1.9403873660829993
```

Before landing, `max_order=860`, 838 problems, order 861 free and the slug unused were confirmed. `collections.json` is registered with minimal intrusion and `todos.md` is updated to 839 problems / audio 13 / next batch from 862. `land861.py` lands 11 files and verifies the three core artifacts against `variants/pytorch/` with `filecmp`.
