# SpecAugment mask_along_axis Segment Generation and Merge Ledger

SpecAugment (paper 1904.08779) zeroes contiguous bands of an acoustic
spectrogram along the time and frequency axes. FunASR's
`funasr/models/specaug/mask_along_axis.py` (espnet-lineage) is the most
common implementation in ASR training pipelines: it randomly samples each
band's width and start, then ORs all candidate boolean rows together. This
problem makes the `(start, width)` pairs explicit inputs and asks you to
replicate the deterministic parts: band generation, boolean-OR merging, and
fill values.

## Interface

```python
solve(spec: list[list[float]], masks: list, axis: str = "time",
      replace_with_zero: bool = True) -> dict
```

- `spec`: a `(T, F)` spectrogram;
- `masks`: list of bands, each `[pos, width]` (the sampled results, made
  explicit);
- `axis`: `"time"` masks whole rows (one value per masked frame) or `"freq"`
  masks whole columns;
- `replace_with_zero`: `True` fills `0.0`; `False` fills the **arithmetic
  mean of every element of the whole spectrogram** (a single scalar,
  computed on the original spec before filling, including the elements
  about to be masked).

Returns:

```python
{"masked": list[list[float]],        # the filled spectrogram (input unchanged)
 "segments": [[start, end], ...]}    # contiguous bands after the OR, ascending
```

## Rules

1. Axis dimension `D = T` (time) or `F` (freq); the official comment states
   `spec_lengths` is unused (no length-aware truncation).
2. Band interval `[pos, pos + width)` is **half-open**:
   `mask[d] = (pos <= d) and (d < pos + width)`. `width == 0` or a fully
   out-of-range band (`pos >= D`) is empty; negative starts clip to 0 and
   the right end clips to D.
3. Multiple bands merge via **boolean OR**: officially each candidate band
   produces a boolean row and `any(dim=1)` combines them. Adjacent bands
   (`[0,2)` and `[2,4)`) and overlapping bands naturally join into one
   contiguous run; `segments` is the run decomposition of that vector.
4. The fill value replaces every masked position; the
   `replace_with_zero=False` mean is a whole-spectrum scalar mean, not a
   per-row/per-column mean.

## Official background (for understanding, not tested)

Official sampling:

```python
mask_length = randint(mask_width_range[0], mask_width_range[1])  # exclusive upper bound
mask_pos = randint(0, max(1, D - mask_length.max()))             # coupled to the batch max width
```

i.e. the start range is governed by the **largest band width in the batch**
(cross-sample coupling), and the width upper bound is exclusive
(`randint(0, 30)` yields 0..29). These random parts are made explicit
inputs here.

## Examples

```python
spec = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]  # T=4, F=3

solve(spec, [[1,2]], "time")
# rows 1 and 2 filled with 0.0
# -> {"masked": [[1,2,3],[0,0,0],[0,0,0],[10,11,12]], "segments": [[1,3]]}

solve(spec, [[0,2],[2,2]], "time")
# adjacent bands join into one run
# -> {"masked": [[0,0,0]]*4, "segments": [[0,4]]}

solve(spec, [[0,1]], "time", replace_with_zero=False)
# whole-spectrum mean = 6.5
# -> {"masked": [[6.5,6.5,6.5],[4,5,6],[7,8,9],[10,11,12]], "segments": [[0,1]]}
```

## Hints

- `segments` is the result of the boolean OR: OR first, then decompose into
  runs — not a per-input-band interval list (though the two coincide
  mathematically). Watch the open right endpoint.
- Duplicate and contained bands do not change the OR; `width=0` bands are
  empty just like out-of-range bands.
- On the freq axis `D = F`: the same band list means different things on the
  two axes (masking rows vs masking columns).
