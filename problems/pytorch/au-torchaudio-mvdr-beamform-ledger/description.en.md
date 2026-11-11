# torchaudio MVDR beamforming ledger: complex PSD outer products with mask normalisation, Tikhonov loading that touches only the real diagonal, Souden complex-trace normalisation, the RTF denominator.real, the rtf_power n_iter schedule, and the conjugate einsum application

## 1. Background

The MVDR block of `torchaudio.functional` (functional.py lines 1799-2198, nine top-level functions) is the last large uncovered mechanism cluster in that file: starting from the cross-channel `psd`, three private helpers (`_compute_mat_trace`, `_tik_reg`, `_assert_psd_matrices`) feed the Souden weights `mvdr_weights_souden`, the explicit-RTF weights `mvdr_weights_rtf`, the power-iteration estimate `rtf_power`, and finally `apply_beamforming` applies the conjugated weights to the multi-channel spectrum. This problem reproduces the whole chain in pure Python and splits it into seven independently judged modes.
Library-wide searches for `mvdr`, `souden`, `rtf_power`, `beamforming`, `diagonal_loading` and `Tikhonov` return no hits. Orders 855, 857, 858, 859, 860 and 861 already covered the spectral post-processing, waveform operators, NCCF, sinc resampling, Griffin-Lim and loudness gating of the same file; this problem uses a disjoint set of functions.

### Official source

```text
pytorch/audio main::src/torchaudio/functional/functional.py
  psd                    (line 1799)
  _compute_mat_trace     (line 1844)
  _tik_reg               (line 1865)
  _assert_psd_matrices   (line 1886)
  mvdr_weights_souden    (line 1913)
  mvdr_weights_rtf       (line 1976)
  rtf_evd                (line 2061, probe only)
  rtf_power              (line 2085)
  apply_beamforming      (line 2148)
```

### Verbatim execution environment

`ast.get_source_segment` extracts the nine definitions byte-for-byte from functional.py and execs them into ONE namespace, so `mvdr_weights_souden` and `rtf_power` resolve the real `_assert_psd_matrices`, `_tik_reg` and `_compute_mat_trace`; no official body is modified. Complex values cross the boundary as nested `[re, im]` lists: `C()` builds complex128 tensors and `cx()` converts back. `rtf_evd` depends on `torch.linalg.eigh`, whose eigenvector sign and phase are LAPACK-implementation-defined, so it is kept OUT of the judged domain and is only probed for its `v[..., -1]` selection.

## 2. Conventions

### Inputs and outputs

Every mode goes through the single entry `solve(mode, **kw)` in pure Python with no torch / torchaudio / numpy / scipy and no global RNG. Complex values are `[re, im]` pairs and real values are Python numbers. The official leading `...` batch axis is stripped, leaving `(freq, channel[, channel])` and `(channel, freq, time)` shapes. Besides the primary result, each mode returns intermediate ledger fields that the judge checks item by item.
```text
mode       primary args                         defaults
psd        specgram, mask=None,                normalize=True, eps=1e-10
trace      mat, dim1=-1, dim2=-2
tik_reg    mat,                                reg=1e-7, eps=1e-8
souden     psd_s, psd_n, reference_channel,    diagonal_loading=True,
                                               diag_eps=1e-7, eps=1e-8
rtf        rtf, psd_n, reference_channel=None, diagonal_loading=True,
                                               diag_eps=1e-7, eps=1e-8
rtf_power  psd_s, psd_n, reference_channel,    n_iter=3,
                                               diagonal_loading=True, diag_eps=1e-7
beamform   weights, specgram
```

### Invalid inputs

The validation order is part of the contract: `_need_cx` checks rank and rectangular leaves first, then squareness / shape agreement, and `pack` converts last. `reference_channel` accepts only an int or a non-bool one-hot list; a bool raises `TypeError('Expected "int" or "Tensor" for reference_channel')` following the official `torch.jit.isinstance` semantics. For `rtf`, a non-square psd_n always hits the `rtf.shape == psd_n.shape[:-1]` guard first, so the dedicated squareness guard is unreachable on that path.

## 3. Mode 1: psd (cross-channel PSD)

`specgram` is a complex `(C, F, T)` spectrum. Official code transposes `(-3, -2)` to `(F, C, T)` and runs the einsum `...ct,...et->...tce` on `t * t.conj()` (the RIGHT operand is conjugated). `mask` is a real `(F, T)` tensor; a mismatch raises `The dimensions of mask except the channel dimension should be the same as specgram`. With `normalize=True` the mask becomes `mask / (mask.sum(-1, keepdim=True) + eps)`, then `mask[..., None, None]` broadcasts into the outer product, and the final sum is over the time axis `dim=-3`. Returns `psd`, `shape`, `mask_norm` and `sum_dim=-3`.

## 4. Mode 2: trace (complex matrix trace)

`_compute_mat_trace` requires `ndim >= 2` and `shape[dim1] == shape[dim2]`, then returns `torch.diagonal(input, 0, dim1, dim2).sum(-1)`. For the judged `(F, C, C)` square matrices both dim orders yield the same diagonal. Returns the complex `trace`, `diag` and `shape`.

## 5. Mode 3: tik_reg (Tikhonov diagonal loading)

The official docstring says it plainly: only the real part is modified. `epsilon = _compute_mat_trace(mat).real[..., None, None] * reg + eps` and `mat + epsilon * eye`; only the real part of the diagonal changes, while off-diagonal entries and diagonal imaginary parts are untouched. `eye` shares the dtype and device of `mat`. Returns `out`, `shape`, `trace_real`, `epsilon`, `eye`, `channels`, `reg` and `eps`.

## 6. Mode 4: souden (Souden MVDR weights)

After `_assert_psd_matrices`, diagonal loading applies `_tik_reg(reg=diag_eps)` to psd_n; `numerator = torch.linalg.solve(psd_n, psd_s)`. Crucially the normaliser is the COMPLEX `_compute_mat_trace(numerator)` with no `.real`: `ws = numerator / (trace[..., None, None] + eps)`. An int reference channel selects column `ws[..., :, rc]`, a one-hot tensor uses `einsum("...c,...c->...", [ws, rc[..., None, None, :]])`. Returns `weights`, `shape`, `branch`, `psd_n_reg`, `numerator`, `trace_numerator` and `ws`. The pure-Python side reproduces `linalg.solve` with complex LU (partial pivoting) plus back substitution; a singular matrix raises `singular matrix in linalg.solve`.

## 7. Mode 5: rtf (MVDR weights from an explicit RTF)

`numerator = solve(psd_n, rtf.unsqueeze(-1)).squeeze(-1)` and `denominator = einsum("...d,...d->...", [rtf.conj(), numerator])` is `v^H @ numerator` (the conjugate is on v, not on the numerator). Crucially the divisor is `denominator.real.unsqueeze(-1) + eps`: the imaginary part of the complex denominator is discarded. When `reference_channel` is not None, a `scale` is applied afterwards: `rtf[..., rc].conj()` for the int branch and `einsum("...c,...c->...", [rtf.conj(), rc[..., None, :]])` for the one-hot branch; None gives `branch="none"` and `scale=None`. Returns `weights`, `shape`, `branch`, `psd_n_reg`, `numerator`, `denominator`, `denominator_real`, `weights_unnorm` and `scale`.

## 8. Mode 6: rtf_power (power-iteration RTF estimate)

`n_iter <= 0` raises `The number of iteration must be greater than 0.`. After loading, `phi = solve(psd_n, psd_s)` (the official comment counts this as the first iteration); the reference channel selects `phi[..., rc]` or an einsum gives `r0`, then `unsqueeze(-1)`. The schedule: for `n_iter >= 2` run `n_iter - 2` steps of `phi @ cur` and finish with one `psd_s @ cur` (`tail="psd_s"`, `n_loop = n_iter - 2`); for `n_iter == 1` do a single `psd_n @ cur` (`tail="psd_n"`, `n_loop=0`), which the comment justifies as `psd_n @ phi @ u = psd_s @ u`. Returns `rtf`, `shape`, `branch`, `psd_n_reg`, `phi`, `rtf_init`, `n_loop`, `tail` and `n_iter`.

## 9. Mode 7: beamform (conjugate einsum application)

`apply_beamforming` is exactly `einsum("...fc,...cft->...ft", [w.conj(), specgram])`: the weights are conjugated, then contracted with the `(C, F, T)` spectrum along the channel axis for each frequency and frame, producing `(F, T)`. With the batch axis stripped, shape validation reduces to matching the channel count C and the frequency count F. Returns `out`, `shape` and `weights_conj`.

## 10. Official quirk list

The `psd` docstring says the eps default is `1e-15` while the signature uses `1e-10`; the `_tik_reg` docstring says reg defaults to `1e-8` while the signature uses `1e-7` (this problem follows the signatures). The `mvdr_weights_rtf` error message contains the official typo `withou`, and the `mvdr_weights_souden` docstring contains `martrix`. The complex-dtype checks of `_assert_psd_matrices` and `rtf` degenerate into `[re, im]` leaf validation in the pure-Python domain. `rtf_evd` takes `v[..., -1]`, but eigenvector phase is not portable, so the whole routine is excluded.

## 11. Example and landing

The verbatim environment passes 241/241 probes in 12 segments, the three-layer cross-check reports 134334 comparisons over 18 configurations ALL GREEN, the judge passes 73/73, and mutation testing catches 128/128 active mutants after excluding 2 documented equivalents. Eleven files are landed (four bilingual docs, meta, solution / starter / test_cases and the variants/pytorch triple), `audio` is registered in `collections.json`, and the short and full guards run with isolated `--basetemp` directories. See `notes.md`.
