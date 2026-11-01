# notes — asr-specaugment-mask (order 693)

## Source

- Official code: modelscope/FunASR `funasr/models/specaug/mask_along_axis.py`,
  function `mask_along_axis` (GitHub `main`, fetched 2026-11; espnet's
  same-named function is the common ancestor, and FunASR's is what Paraformer
  training actually uses). The sibling `specaug.py` composes T/F axes with
  time_warp.
- Paper background: SpecAugment 1904.08779 (Park et al.) defines mT/mF
  spectral masking; FunASR's variant ignores `spec_lengths` and adds an LFR
  variant (`mask_along_axis_lfr`) repeating bands per `lfr_rate` — LFR is
  not covered here.
- Scoping decision: torchaudio's `FrequencyMasking/TimeMasking` is another
  common source, but the pytorch/audio repo moved its Python side away after
  v2.9 (contents API 404), and in the ASR-collection context FunASR is the
  better fit and pure-numpy transcribable — FunASR was chosen.

## Ledger rules

1. `D = spec.shape[dim]` (time=1 → T, freq=2 → F); single-sequence view.
2. Band boolean: `(mask_pos <= aran) & (aran < mask_pos + mask_length)`,
   half-open; width=0 → empty; out-of-range clips naturally because aran
   stops at D-1.
3. `mask.any(dim=1)` boolean OR — **adjacent bands ([0,2)+[2,4)) join into
   one run on the boolean vector**. This is the mechanism behind the todos'
   "adjacent band merge rule": not special-case logic, just the natural
   result of any. The `segments` output is the run decomposition of that
   vector.
4. `replace_with_zero=False` → `value = spec.mean()`: a whole-spectrum
   scalar (including elements about to be masked, computed on the original
   spec before masked_fill). Officially `masked_fill_` is in-place when
   `requires_grad=False` — no observable difference for a single sequence.
5. Official sampling quirks (made explicit in the description background):
   - width `randint(lo, hi)` has an exclusive upper bound (0..29);
   - start `randint(0, max(1, D - mask_length.max()))` uses the **batch-wide
     max width** for its cap — cross-sample coupling, which from a
     single-sample view means narrow bands can start beyond
     `D - own_width` (guaranteed truncation). A real counter-intuitive
     point of the official implementation.
   - The `MaskAlongAxisLFR` lfr-repeat variant is not covered.

## Oracle design

`_oracle` uses classic interval algorithms: clip bands to [0, D] (width<=0
dropped) → sort → interval merge → expand. Compared against solve
(per-frame boolean + scan) as two independent paths on 300 random cases
(D 1..12, negative starts / out-of-range bands / width=0, both
replace_with_zero settings), plus assertions that segments ascend, are
disjoint, half-open, and in bounds.

## Mutation record

13 mutations: 12 caught + 1 equivalent. The equivalent
`skip-masked-copy` (`row[:]` → `list(row)`, identical row-copy semantics)
is analyzed here. The 12 observable ones cover: both endpoint
open/closed flips, width shift, axis selection, time-axis filling by
columns, mean timing (after filling), mean granularity (per-row), segment
right endpoint, adjacent-run non-merge, segment ordering, and
ignore-mask fill-all.

## Dedup

- The `audio` frontend collection's five problems are signal processing
  (stft/frame etc.); no augmentation problem existed — this is the repo's
  first augmentation ledger (confirmed against the todos red lines).
- Different from `dt-*` token masks (attention/padding semantics): this is
  spectral band augmentation.
- time_warp (the other FunASR specaug component, a DTW-style elastic
  deformation) is heavier; if it ever becomes a problem it needs its own
  dedup pass.
