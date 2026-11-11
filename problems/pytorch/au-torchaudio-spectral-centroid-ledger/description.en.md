# torchaudio spectral_centroid ledger: the linspace(0, sample_rate//2, 1+n_fft//2) frequency grid, the (-1,1) column-vector broadcast, the RAW-specgram sum on dim=-2 as denominator, 0/0 -> nan, and the exact spectrogram kwargs

## 1. Background

`torchaudio.functional.spectral_centroid` computes a per-frame spectral centroid from a magnitude spectrogram and a linear frequency grid. It has NO RNG, NO audio IO and NO warning: it only calls `spectrogram` for the magnitude spectrogram, one `linspace` and two `sum(dim=-2)` reductions. This problem reproduces it in pure Python as eight independently judged modes (`centroid` / `freqs` / `call_kwargs` / `ledger` / `zero_frame` / `shapes` / `errors` / `census`) and pins every value to verbatim official snapshots.
Library-wide duplicate check: `spectral_centroid` has only TWO hits across every `meta.json`, `description*.md` and `notes*.md` under `problems`, and both are the "zero-hit avoidance" exclusion notes in orders 859 / 860 rather than actual coverage; those historical notes explicitly list it as a later candidate. Order 855 extracts `spectrogram` itself plus four post-processing functions; this problem does NOT re-judge `spectrogram`: it is injected as an explicit input spy, and the judge targets only the exact call kwargs and the centroid's own semantics, giving a clean boundary.

### Official source

```text
pytorch/audio main::src/torchaudio/functional/functional.py
  spectral_centroid(waveform, sample_rate, pad, window, n_fft,
                    hop_length, win_length)   (lines 1257-1299)

  specgram = spectrogram(waveform, pad, window, n_fft, hop_length,
                         win_length, power=1.0, normalized=False)
  freqs = torch.linspace(0, sample_rate // 2, 1 + n_fft // 2,
                         device=specgram.device).reshape((-1, 1))
  freq_dim = -2
  return (freqs * specgram).sum(dim=freq_dim) / specgram.sum(dim=freq_dim)
```

`official868.py` extracts the top-level `spectral_centroid` with `ast.get_source_segment` and execs it into one shared namespace; NO official body is modified. The only injected objects are real `torch`, `math`, `typing`, `warnings` and a `spectrogram` spy. The spy returns an explicitly supplied magnitude spectrogram and records the exact call kwargs -- the same technique as order 855 (stft spy) and 864 (STFT / ISTFT / phase_vocoder injection). The global default dtype is pinned to `float64`, so torch's `linspace`, broadcast and division can be cross-checked bit-for-bit against the pure-Python reference.

## 2. Conventions

### Eight modes

`solve(mode, **kw)` dispatches eight ways: `centroid` returns per-frame centroids, `freqs` returns the `(F,1)` column-vector frequency grid, `call_kwargs` returns the 12 kwargs recorded by the spy, `ledger` returns the full ledger (grid / weighted matrix / numerator / denominator / output shape / call), `zero_frame` judges the `nan` from an all-zero frame, `shapes` judges 2-D / 3-D / 4-D inputs, `errors` judges six boundary cases, and `census` judges source-level facts. Apart from `unknown mode` raising `ValueError`, every exception is folded into a `{type, msg}` dict whose text matches the official wording verbatim.

### Input domain

The judge constrains `n_fft >= 2` with `1 + n_fft // 2` equal to the specgram's frequency dimension `F`. **`n_fft <= 1` is explicitly outside the input domain**: there `torch.linspace(..., steps=1)` yields a single-bin grid that triggers PyTorch's `(1,1)` broadcast (centroid is always 0), while `steps=0` yields an empty tensor and a different broadcast error path; both are torch tensor implementation details rather than the centroid algorithm itself. The `freqs` mode still exports the official `n_fft=0` constant because that mode never goes through the broadcast.

## 3. Five official quirks

(1) `power=1.0` (magnitude spectrogram) and `normalized=False` are the ONLY explicitly passed kwargs; `center` / `pad_mode` / `onesided` / `return_complex` all take the spectrogram defaults (`True` / `"reflect"` / `True` / `None`), so passed and defaulted arguments must be judged separately. (2) `sample_rate // 2` is floor division: 16000 and 16001 both give 8000, and a negative sample rate follows Python floor semantics (`-8000 // 2 = -4000`).
(3) the grid has `1 + n_fft // 2` bins (not `n_fft // 2`), and the back half of `linspace` is computed backwards from `end`; the pure-Python reference must reproduce this symmetric formula for bit-for-bit agreement. (4) the DENOMINATOR sums the RAW `specgram`, not the broadcast result: `specgram.sum(dim=-2)` needs at least 2 dims, so a 1-D spectrogram raises `IndexError ... got -2` even though the numerator would broadcast fine.
(5) an all-zero frame is `0/0`, whose official output is `nan` (normalized to the string `"nan"` in the judge); a zero denominator with a NONZERO weighted numerator gives IEEE `+inf` / `-inf`. The output shape is `shape[:-2] + shape[-1:]`, dropping the frequency dimension while keeping leading dims and the time dimension.

## 4. Judge ledger

`fixtures868.py` exports 33 constants from the verbatim official environment, and `cases868_checks.py` supplies 50 `check_*(solve)` functions: floor-division and symmetric-formula checks on the frequency grid, all 12 spy kwargs, the full ledger intermediates, 2-D / 3-D / 4-D shapes, six error dicts, the 14 source-level census facts, the complete `unknown mode` message, plus three hardening checks -- a fresh dict from every `census` call, the `0/0` and `+/-inf` branches, and all three `shapes(n_fft=2)` layouts returning real error dicts instead of a swallowed one.

## 5. Verification pipeline

```text
official868.py smoke        8 modes, EXIT=0
probe868.py                 117/117 ALL GREEN (12 segments)
probe_solution868.py        11769 checks / 68 configs / 0 bad
verify868.py                43488 checks / 84 configs / 99 props / 0 fails
fixtures868.py              33 constants
driver868.py                50/50 ALL ACCEPTED
mut868.py                   155 mutants / 4 proven equivalents / 151 CAUGHT
equiv868.py                 4 mutants x 1692 comparisons = 6768, 0 diffs
```

The third cross-check layer shares NO algorithm with the reference: it recomputes the frequency grid and the weighted sums with exact `Fraction` rationals, then validates formula-independent algebraic properties -- a single-bin impulse spectrum must have its centroid at exactly that bin's frequency, the centroid must lie in `[f_min, f_max]`, and scaling one frequency column by `s` scales numerator and denominator alike leaving the centroid unchanged. All 50 judge checks are driven by official constants.

## 6. Mutation and equivalence proofs

The first mutation round produced 8 SURVIVED and 5 ANCHOR-ERRORs out of 155 mutants. The 5 anchor errors came from `c["sample_rate"] // 2` and `1 + c["n_fft"] // 2` each appearing twice (once in `_freq_vec`, once in `_ledger`); explicit occurrence indices fixed them. Of the 8 survivors, 4 were JUDGE BLIND SPOTS and were caught by new hardening cases: `census` returning a shared object, the `+/-inf` branch at a zero denominator, and the swallowed exception branch in `shapes`.
The other 4 are GENUINE SEMANTIC EQUIVALENTS, proven by `equiv868.py` comparing FULL `solve()` output (including exception type and message) over 188 in-scope configurations x 9 modes -- 6768 comparisons, zero differences: `e-linspace-msg` / `l-steps-neg` touch the unreachable `steps<=0` branch when `n_fft>=2`; `l-step-num` is exact because `start` is always 0 so `end-start == end+start`; `er-odd-sr` yields identical output since `16001//2 == 16000//2`. NO mutant is deleted; the final 151 active mutants are all CAUGHT with 0 SURVIVED and 0 ERROR.

## 7. Registration and guards

`register868.py` registers with minimal intrusion, growing `audio` from 19 to 20 and appending the 868 convention paragraph to both descriptions; verification expects `846 registered = 846 disk / max_order=868 / no dangling / no unregistered`. `todos.md` is updated to 846 problems / audio 20 / next batch from 869 and flips 868 from planned to landed. After landing, selftest runs in both `submit` and `run` modes, followed by the library-wide bilingual check, the short guard and a full guard with an isolated `--basetemp`. Per the standing convention, NO git commit is made until the user confirms.
