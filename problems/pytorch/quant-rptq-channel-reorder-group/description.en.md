# Quantization: RPTQ Channel Reorder and Group Quantization

## Problem

Adjacent channels often have very different activation ranges. If channels are grouped in their natural order, one outlier channel stretches the group range and the remaining channels get very few integer grid points. RPTQ (Yuan et al., ICLR 2023, arXiv:2304.01089) first reorders channels by their **quantization span** so that channels with similar spans land in the same group, then quantizes each group.

Given per-channel activation bounds `xmax/xmin` and a weight matrix `W`, implement:

1. Compute `span_i = xmax_i - xmin_i` and obtain `perm` by a **stable** ascending sort on `(span_i, channel index)`;
2. Split the `C` channels in `perm` order into `num_groups` **balanced** segments: the first `C % num_groups` segments get one extra channel; write the segment lengths to `counts`;
3. For each segment, take its `max/min` and compute the group `scale` and `zero`:
   - `scale = clamp((gmax - gmin) * 2^-bits, 1e-4, 1e4)`
   - `zero = clamp(round((gmax + gmin) * -0.5 / scale), -1e4, 1e4)`
4. Fake-quantize each element of `W`: `q = clamp(round(w / scale) + zero, -2^(bits-1), 2^(bits-1)-1) - zero`, reconstruct `w_hat = q * scale`, then write values back to the original channel order according to `perm`.

The integer code is clamped to the **symmetric** interval `[-2^(b-1), 2^(b-1)-1]`. The positive side has only `2^(b-1)-1` codes, so the group's positive maximum usually saturates first.

## Task

Implement `solve(xmax, xmin, W, num_groups, bits)`:

- `xmax`, `xmin`: float32 `(C,)` per-channel activation upper/lower bounds, requiring `xmin <= xmax`;
- `W`: float32 `(O, C)` weights;
- `num_groups`: an integer in `1..C`; `bits`: an integer in `2..8`.
- Return a `dict`: `perm` int64 `(C,)`, `counts` int64 `(G,)`, `scales` float32 `(G,)`, `zero_points` int64 `(G,)`, `dequant` float32 `(O, C)`.

**Conventions**: compute internally in float64 and return the dtypes above; tolerance `1e-5`. Invalid inputs (wrong types, shape mismatch, out-of-range `num_groups/bits`, non-finite values, `xmin > xmax`) raise `ValueError`.

## Example

```
xmax = [8.0, 0.5, 8.4, 0.6], xmin = [-8.0, -0.5, -8.4, -0.6]
W = [[8.0, 0.5, 8.4, 0.6], [-8.0, -0.5, -8.4, -0.6]]
num_groups = 2, bits = 4

span = [16.0, 1.0, 16.8, 1.2] -> perm = [1, 3, 0, 2], counts = [2, 2]
group 0: channels {1, 3}, gmax=0.6, gmin=-0.6 -> scale=0.075, zero=0
group 1: channels {0, 2}, gmax=8.4, gmin=-8.4 -> scale=1.05, zero=0
dequant = [[7.35, 0.525, 7.35, 0.525], [-8.4, -0.525, -8.4, -0.6]]
```

## Constraints

- The sort must be stable: ties on span are broken by ascending channel index; do not rely on float-sensitive clustering.
- Segments must be balanced, with the remainder assigned to the first `C % num_groups` segments (the upstream PEG variant drops it; this problem keeps every channel).
- Each group may only use the `max/min` of its own segment, never global statistics.
- Write reconstructed values back to the original channel order according to `perm`.
