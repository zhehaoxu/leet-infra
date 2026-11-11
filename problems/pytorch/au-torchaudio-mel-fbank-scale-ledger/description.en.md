# torchaudio mel / linear filterbank scale ledger: the HTK and Slaney scales, a tensor boolean mask, the n_mels+2 point grid and the triangular slope form, two independent norm and mel_scale switches, and the all-zero-column warning

## 1. Background

`_hz_to_mel`, `_mel_to_hz`, `_create_triangular_filterbank`, `melscale_fbanks` and `linear_fbanks` in `torchaudio.functional` form the mel and linear filterbank construction chain. They only perform scale conversion, `linspace` grids, triangular weights and area normalization: no STFT, no audio IO, no model forward, and NO RNG in any of the five bodies. This problem reproduces them in pure Python as ten independently judged modes and pins every number to verbatim official snapshots.
Library-wide duplicate check: the five function names have ZERO hits across every `meta.json`, `description*.md` and `notes*.md` under `problems`. Order 342 is a legacy mel filterbank problem that BIND the Slaney scale and area normalization into one convention; order 852 is the Kaldi convention (`1127*ln(1+f/700)`) plus a VTLN / framing / mfcc compatibility layer. Order 866 judges the scale intermediates and triangular matrices of the five official functions directly, covering `linear_fbanks` being equally spaced in Hz, the zero-column warning and the orthogonality of `norm` and `mel_scale`; the boundary with both earlier problems is explicit.

### Official source

```text
pytorch/audio main::src/torchaudio/functional/functional.py
  _hz_to_mel                    (lines 425-456)
  _mel_to_hz                    (lines 459-489)
  _create_triangular_filterbank (lines 492-515)
  melscale_fbanks               (lines 518-587)
  linear_fbanks                 (lines 590-633)

  _hz_to_mel / htk:
    return 2595.0 * math.log10(1.0 + (freq / 700.0))
  _hz_to_mel / slaney:
    f_sp = 200.0 / 3 ; min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0
    mels = (freq - f_min) / f_sp
    if freq >= min_log_hz:  # scalar if, NOT a tensor mask
      mels = min_log_mel + math.log(freq / min_log_hz) / logstep

  _mel_to_hz / slaney:
    freqs = f_min + f_sp * mels
    log_t = mels >= min_log_mel        # TENSOR boolean mask
    freqs[log_t] = min_log_hz * torch.exp(logstep * (mels[log_t] - min_log_mel))

  _create_triangular_filterbank:
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]
    up_slopes   = slopes[:, 2:] / f_diff[1:]
    fb = torch.max(zero, torch.min(down_slopes, up_slopes))

  melscale_fbanks:
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    f_pts = _mel_to_hz(m_pts, mel_scale=mel_scale)
    enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
    if (fb.max(dim=0).values == 0.0).any(): warnings.warn(...)

  linear_fbanks:
    f_pts = torch.linspace(f_min, f_max, n_filter + 2)
    # no norm, no mel_scale, no warning
```

`official866.py` extracts the five top-level functions with `ast.get_source_segment` and execs them into one shared namespace; NO official body is modified. The only injected objects are real `torch`, `math`, `warnings` and type aliases, with the global default dtype pinned to `float64` so the `linspace` / `pow` / `exp` chain can be cross-checked against a pure-Python reference. The ledger layer (all_freqs / m_pts / f_pts / f_diff / enorm / zero_columns / column_max / warnings) is recomputed with real torch in `official866.py` and is the sole authority for judge constants.

## 2. Conventions

### Ten modes

`solve(mode, **kw)` dispatches ten ways: `hz_to_mel` / `mel_to_hz` are single-value scales, `mel_grid` / `hz_grid` are multi-value scales, `triangular` judges the slope-form matrix directly, `mel_fbanks` / `linear_fbanks` judge full matrices, `fbank_ledger` judges every intermediate, `errors` judges whether seven configurations raise, and `census` judges source-level facts. Apart from `unknown mode` raising `ValueError`, every exception is folded into a `{type, msg}` dict whose text matches the official wording verbatim.

### Scales and grids

HTK is `2595*log10(1+f/700)` with inverse `700*(10**(m/2595)-1)`. Slaney splits at 1000 Hz: `f/(200/3)` below and `min_log_mel + log(f/1000)/logstep` above, where `logstep=log(6.4)/27` makes 6400 Hz exactly 27 mel above the break; the inverse uses a TENSOR boolean mask `log_t = mels >= min_log_mel` with in-place assignment, not a scalar if. `min_log_mel = 1000/(200/3) = 14.999999999999998` (not exactly 15), so 15.0 lands in the log branch.
The mel grid is `n_mels+2` mel points from `linspace` inverse-mapped to `f_pts`; `all_freqs = linspace(0, sample_rate // 2, n_freqs)` uses floor division, so 16001 and 16000 share the same endpoint. The triangular matrix uses SHIFTED denominators: down slopes divide by `f_diff[:-1]`, up slopes by `f_diff[1:]`, then `max(0, min(down, up))` clamps to nonnegative.

## 3. Five official quirks

(1) `norm` and `mel_scale` are TWO INDEPENDENT switches: `norm="slaney"` only multiplies columns by `enorm=2/(f_pts[j+2]-f_pts[j])` and leaves the scale grid untouched; all four combinations must be pairwise distinct. (2) `linear_fbanks` spaces its points equally in Hz, has no `norm` / `mel_scale` in its signature and NEVER warns even when many columns are all zero; an invalid `mel_scale` is silently ignored.
(3) The all-zero-column warning is evaluated AFTER `fb *= enorm`, but `enorm` is positive so a zero column cannot be resurrected; the text interpolates both `n_mels` and `n_freqs` and must be preserved verbatim. (4) `n_mels=0` degenerates to one empty list per row with no exception and no warning; `f_min=-1` is not validated, so the HTK `log10` stays legal and yields a negative mel. (5) `norm` accepts only `None` / `"slaney"` and `mel_scale` only `"htk"` / `"slaney"`, both case-sensitive.

## 4. Judge ledger

`fixtures866.py` exports 121 constants from the verbatim official environment, and `cases866_checks.py` supplies 123 `check_*(solve)` functions: numeric tolerance comparisons plus BIT-FOR-BIT EXACT equality assertions for `torch.linspace(float64)` (the reference reproduces PyTorch's symmetric formula, `i<steps//2 -> start+i*step` and otherwise `end-(steps-1-i)*step`, measured to be identical to torch). Hardening also covers exception types, warning text, leaves being Python `float`s, all four `norm` x `mel_scale` combinations being pairwise distinct, and end-to-end recomputation of full matrices via `triangular`.

## 5. Verification pipeline

```text
official866.py smoke          10 modes, EXIT=0
probe866.py                   821/821 ALL GREEN (12 segments)
precision866.py               symmetric linspace diff = 0
probe_solution866.py          6344 good / 0 bad
verify866.py                  37578 checks / 69 configs / 0 fails
fixtures866.py                121 constants
driver866.py                  123/123 ALL ACCEPTED
equiv866.py                   8 mutants x 2700 comparisons / 0 diffs
mut866.py                     168 mutants - 8 equivalents = 160 CAUGHT
```

The third cross-check layer shares NO algorithm with the reference: `fractions.Fraction` for exact rational `linspace`, `log1p` / `expm1` / `pow(6.4, x)` rewrites for both scales, a geometric piecewise definition (rising on the left half, falling on the right) for the triangular matrix, and set membership for all-zero columns. The eight mutants claimed equivalent (the unreachable `steps<=0` branch, two threshold comparisons whose branches are numerically identical at the break point, an unconsumed third return value, `[0]` vs `[-1]` on a single-element list, and two `errors` probe configurations that do not change the result) are each proved by `equiv866.py` over 270 configurations x 10 modes; no mutant is ever deleted just to make the suite green.

## 6. Engineering lessons and scope cuts

The first trap is the SYMMETRIC BACK HALF of `torch.linspace`: a naive `start+i*step` drifts by about 9e-13 on an 8-point grid, entirely invisible under a 1e-8 tolerance, so an exact `==` assertion is required to kill the mutant that flattens the symmetric formula. The second trap is that `>=` and `>` in the `mel_to_hz` tensor mask produce THE SAME DOUBLE at `mel == min_log_mel` (`1000*exp(0)` and `f_sp*min_log_mel` are both 1000.0), and likewise both `hz_to_mel` branches give 14.999999999999998 at 1000 Hz; those two mutants are genuine equivalents rather than blind spots.
The third trap is `n_freqs=0`: the official `melscale_fbanks` raises `IndexError` from `fb.max(dim=0)`, while the `linear_fbanks` exception comes from the reference environment's ledger helper -- a torch implementation detail rather than problem semantics -- so it is explicitly outside the input domain (`n_freqs>=1`, `n_mels>=0`). Conversely, `linear_fbanks` returning a normal matrix for an invalid `mel_scale` and staying silent with all-zero columns were added as hardening cases.

## 7. Registration and guards

`register866.py` registers with minimal intrusion, growing `audio` from 17 to 18 and appending the 866 convention paragraph to both descriptions; verification expects `844 registered = 844 disk / max_order=866 / no dangling / no unregistered`. `todos.md` is updated to 844 problems / audio 18 / next batch from 867 and flips 866 from planned to landed. After landing, selftest runs in both `submit` and `run` modes, followed by the library-wide bilingual check, the short guard and a full guard with an isolated `--basetemp`. Per the standing convention, NO git commit is made until the user confirms.
