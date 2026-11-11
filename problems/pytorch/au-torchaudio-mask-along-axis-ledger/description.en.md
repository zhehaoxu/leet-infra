# torchaudio SpecAugment mask ledger: the `p==1.0` shortcut and `int` truncation, reshape packing and the dead width guard in the shared path, two transposes and split casts in the per-example path, and the `where`/`repeat` branch for a Tensor `mask_value`

## 1. Background

`_get_mask_param`, `mask_along_axis` and `mask_along_axis_iid` in `torchaudio.functional` implement the frequency and time masks of SpecAugment. They only resolve parameters, cast intervals and broadcast a mask: there is no spectral computation or model forward, and all randomness comes from two `torch.rand` calls. This problem reproduces the three official functions in pure Python as nine independently judged modes, replacing the `torch.rand` draws with explicit inputs, so the judged domain contains no RNG at all.
Library-wide duplicate check: order 693 was previously recorded as covering `mask_along_axis`, but a source revisit showed that 693 is a FunASR/espnet EXPLICIT SEGMENT LIST ledger and has nothing to do with these three torchaudio functions. Orders 855 (spectrogram / DB / deltas / CMN), 857 (phase vocoder), 859 (sinc kernel), 860 (Griffin-Lim / ISTFT), 861 (loudness / biquad), 862 (MVDR), 863 (FID) and 864 (speed / pitch-shift orchestration) have disjoint function sets.

### Official source

```text
pytorch/audio main::src/torchaudio/functional/functional.py
  _get_mask_param        (line 806,  5 lines)
  mask_along_axis_iid    (line 813, 70 lines)
  mask_along_axis        (line 885, 74 lines)

  _get_mask_param:
    if p == 1.0: return mask_param
    else:        return min(mask_param, int(axis_length * p))

  shared / mask_along_axis:
    specgram = specgram.reshape([-1] + list(shape[-2:]))
    value = torch.rand(1) * mask_param
    min_value = torch.rand(1) * (size - value)
    mask_start = (min_value.long()).squeeze()
    mask_end = (min_value.long() + value.long()).squeeze()
    if mask_end - mask_start >= mask_param: raise ValueError(...)

  iid / mask_along_axis_iid:
    value = torch.rand(shape[:dim-2], dtype=dtype) * mask_param
    mask_end = (min_value.long() + value.long())[..., None, None]
    specgrams = specgrams.transpose(axis, -1)
    torch.where(mask, mask_value.repeat(specgrams.shape), specgrams)
    specgrams = specgrams.transpose(axis, -1)
```

`official865.py` extracts the three top-level functions with `ast.get_source_segment` and execs them into one shared namespace; NO official body is modified. The only injected object is a forwarding `torch` proxy that replaces `rand` with a recorder returning explicit caller-supplied tensors; every other attribute (`arange`, `where`, `masked_fill`, `transpose`, `reshape`, dtypes ...) is real torch and `Tensor` stays the real `torch.Tensor`, so the `isinstance(mask_value, Tensor)` branch remains live.

## 2. Conventions

### Inputs

The spectrogram is a rectangular nested numeric list (int / float leaves are legal, bool and string leaves are rejected, tuples are accepted too); `variant == "iid"` selects `mask_along_axis_iid` and ANY other string falls through to `mask_along_axis`. A None `axis` resolves to `dim-1` (time); only `dim-2` (frequency) and `dim-1` (time) are accepted and negative axes are rejected. `mask_param`, `mask_value`, `p`, `dtype`, `value_draws` and `start_draws` are all explicit inputs; defaults are a `(2,3,4)` spectrum, `mask_param=3`, `mask_value=0.0`, `p=1.0` and `dtype=float64`.
`value_draws` / `start_draws` must match the recorded `torch.rand` shape element for element: the shared variant always uses shape `[1]`, while the iid variant uses the leading dims `shape[:-2]` of the spectrum; a mismatch returns the verbatim error message. The shared `torch.rand(1)` has NO dtype argument, so its interval arithmetic runs in the global default `float32` even for a `float64` spectrum; the iid variant passes `dtype=waveform.dtype` and therefore follows the spectrum dtype. The pure-Python reference quantizes each scalar step with `struct.pack/unpack("f")` to match float32 bit for bit.

### Nine modes

```text
mode            return value
--------------  ------------------------------------------------------
param           _get_mask_param(mask_param, p, sh[axis])
rand_calls      the two recorded torch.rand calls (shape/dtype/values)
interval        {value, min_value, start, end, width} per example
mask            half-open 0/1 row, nested over the leading dims
output          masked spectrogram, or the verbatim error dict
identity        {same_object, rand_calls, err}
errors          fixed invalid configurations -> verbatim messages
tensor_value    iid path with a real Tensor mask_value (repeat/where)
census          source-level facts about the three official bodies
```

`interval`, `mask` and `output` return `None`, `[]` and the original spectrum respectively in the `mask_param < 1` shortcut; `identity.same_object` is `True` there and `False` otherwise. `output` and `tensor_value` return a `{"type", "msg"}` dict instead of raising on an error; an unknown mode raises `ValueError` whose message must list all nine names in the original `MODES` order.

## 3. Official mechanics, section by section

### 3.1 The `p==1.0` shortcut and `int` truncation

`_get_mask_param` returns `mask_param` DIRECTLY when `p == 1.0`, never looking at the axis length; otherwise it returns `min(mask_param, int(axis_length * p))`. Note that `int()` truncates toward zero while both official docstrings say `floor(...)`: for a `(2,3,4)` spectrum, `p=0.5` gives 2 on the time axis (length 4) and 1 on the frequency axis (length 3); `p=0.999` gives 3 and `p=0.39` gives 1. The length must be `sh[axis]`, not `sh[-1]`, or the frequency axis with `p<1` is wrong.
`p` is validated on the CLOSED interval `[0.0, 1.0]` and both endpoints are accepted; `p=0` makes `mask_param` zero and hits the `mask_param < 1` shortcut, which returns the ORIGINAL object (no new tensor, no rand draw consumed). The `param` mode calls the function directly, so a negative `p` can distinguish `int(-1.2) == -1` from `floor(-1.2) == -2`.

### 3.2 Reshape packing and the dead width guard in the shared path

`mask_along_axis` first packs arbitrary leading dims with `reshape([-1] + shape[-2:])` into 3-D and later restores them with `reshape(shape[:-2] + shape[-2:])`, so ALL examples share ONE mask interval. The mask index is remapped to `axis - dim + 3`; on the frequency axis an extra `mask.unsqueeze(-1)` is needed to broadcast over time. The two `torch.rand(1)` calls sample `value` (the width) and `min_value` (the upper bound for the start).
The endpoints are `mask_start = min_value.long().squeeze()` and `mask_end = (min_value.long() + value.long()).squeeze()`: the two `.long()` casts are SEPARATE, so `end = trunc(min_value) + trunc(value)` rather than `trunc(min_value + value)`. When the fractional parts sum above 1 the two differ by one (a configuration with `value = 0.27 * 10 = 2.7` and a `start_draw` of `0.615384615...` pins `end=2` while a combined cast would give 3). The comparison is the half-open interval `start <= d < end`.
Immediately afterwards comes `if mask_end - mask_start >= mask_param: raise ValueError("Number of columns to be masked should be less than mask_param")`. Since `value < mask_param` and separate casts only shrink the width, this guard is UNREACHABLE under legal draws and requires an injected `value_draws > 1.0` to fire; it is official dead code, but its message must be preserved verbatim. The iid variant has NO such guard and happily produces an over-wide injected mask.

### 3.3 Two transposes and the Tensor branch in the iid path

`mask_along_axis_iid` requires at least three dimensions and draws `value` / `min_value` once per leading-dims shape `shape[:dim-2]` (with `dtype=specgrams.dtype`), then appends `[..., None, None]` for broadcasting. It transposes `axis` with `-1` so the masked axis is last, applies the mask, then transposes back; `mask` comes from `torch.arange(0, size(axis))`.
When `mask_value` is a real `Tensor` the code uses `torch.where(cond, mask_value.repeat(specgrams.shape), specgrams)` -- note that `repeat` receives the shape AFTER the transpose; otherwise it uses `masked_fill(cond, mask_value)`. A scalar tensor and a float produce identical output, but a non-scalar tensor fails because the repeated size does not match. The `tensor_value` mode forces the iid function and wraps `mask_value` into a real tensor to pin this branch.

### 3.4 Verbatim dim / axis / p guards

The shared path requires `dim >= 2` (message mentions `time and frequency`) and the iid path requires `dim >= 3`; both report the NUMBER OF DIMENSIONS in `(%d given)`, not an axis length. The legal axis set is `[dim-2, dim-1]`; out-of-range and negative axes share `Only Frequency and Time masking are supported (axis %d and axis %d supported; %d given).`, and an out-of-range `p` reports `The value of p must be between 0.0 and 1.0 (%s given).`. All these strings come from the verbatim official environment.

## 4. Code skeleton

```python
def solve(mode, **kw):
    c = _cfg(kw)                 # explicit draws, rectangular spec
    r = _run(c)                  # guards -> rand calls -> interval
    if mode == "param":     return r["param"]
    if mode == "rand_calls": return r["calls"]
    if mode == "interval":   return r["iv"]
    if mode == "mask":       return r["rows"]
    if mode == "output":     return r["err"] or r["out"]
    if mode == "identity":   return r["identity"]
    if mode == "tensor_value": ...   # force iid + real Tensor
    if mode == "errors":     ...     # fixed invalid configs
    if mode == "census":     ...     # source-level facts
    raise ValueError("unknown mode: ... (expected one of ...)")
```

The reference uses only `math` and `struct`: rectangular validation and reshaping use hand-written `_shape` / `_nest` / `_map1` / `_map2`, float32 is quantized per scalar step with `struct`, interval endpoints use `math.trunc` (NOT `floor`), and mask application uses a recursive `_apply`. The judge cases contain no torch, never import the reference and use no global RNG; every expected value is a constant snapshot dumped from the official environment.

## 5. Key quirks and traps

1. **The docstring says floor but the code uses `int()`**: `int` truncates toward zero, agreeing with floor on the legal non-negative domain, but a negative `p` in the `param` mode separates them. 2. **The shared path is float32**: `torch.rand(1)` has no dtype, so interval arithmetic uses the global default `float32`; the iid path passes `dtype=specgrams.dtype`, so an iid run with `dtype=float32` and a shared run disagree on `min_value` for the same draw. 3. **Separate `.long()` casts**: `end` adds two independently truncated values, not one truncation of their sum.
4. **The dead width guard is shared-only**: injecting `value_draws=1.5` makes the shared path raise while iid succeeds. 5. **`mask_param<1` returns the original object**: no rand draw is consumed and `identity.same_object` is `True`. 6. **The Tensor `mask_value.repeat` happens AFTER the transpose** and only the iid path has that branch. 7. **Asymmetric leading dims**: for shape `(2,3,3,4)` the `_map2` slice step is `prod(shape[1:])=3`; using `prod(shape[:-1])=2` misaligns rows, and only with n>=5 do the default draws cross 1.0 so that the `% 1.0` modulo becomes observable.

## 6. Verification pipeline

Twelve official-mechanism probe segments give `380/380 ALL GREEN` (verbatim source comparison, the `p` shortcut and truncation, dim / axis / p messages, rand shape and dtype, shared reshape packing, separate casts, the dead guard, iid double transpose, Tensor repeat / where, negative spectra and a single time step, census, and end-to-end ledger consistency). The three-layer cross-check reports `22054 checks / 74 configs / 0 fails`: official runs the real verbatim environment, the reference uses pure-Python recursion with `struct` quantization, and the independent oracle uses flat-index arithmetic, set membership, `Decimal ROUND_DOWN` truncation and `ctypes.c_float` quantization; no algorithm is shared.
The judge driver reports `140/140 ALL ACCEPTED`. For the 169 mutants, thirteen hardening cases (MODES order in the unknown-mode message, forwarding the identity error dict, the `[0]` shape of empty spectra / draws, rejecting a string spec, modulo and slice step under `(2,3)` leading dims, trunc with a negative `p`, a frequency-axis `p=0.5` interval, a nonzero fill value, and the float TYPE of an integer fill value) reduced the survivors to three; `equiv865.py` then proved each equivalent over 1901 configurations x 10 modes. Final result: **169 mutants, 3 documented equivalents excluded, 166/166 CAUGHT, 0 SURVIVED, 0 ERROR**.

## 7. Scope cuts and registration

The judged domain excludes real audio IO, mel / STFT numerics, model forwards and training; the spectrum, `torch.rand` draws and `mask_value` are all explicit inputs, pure Python with no numpy / torch / scipy and no global RNG. The shared and per-example paths must be accounted for separately: never copy the shared width guard into iid, and never write `int()` as `floor()`. After registration `audio` grows 16 -> 17, the library 842 -> 843 and the order pool reaches 865; the short guard uses an isolated `--basetemp` and the full guard result is backfilled after its background run. Per the standing convention, no git commit is made until the user confirms.
