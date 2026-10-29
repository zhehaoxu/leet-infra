# Quantization: AWQ's Joint Scale-and-Clip Grid Search

## Problem

AWQ's per-input-channel scale \(s_j\) is not the only knob. **Clipping** is the second: a few weight outliers inflate a group's quantization scale until every other weight in the group rounds to 0; clipping those outliers to a threshold trades a little magnitude for a group full of usable codes. The **shrink bound** is the third: if the scale squeezes a channel's weights too hard, their own quantization error comes back, so \(s_j\) needs a lower bound.

Given weights, calibration activations and a grid over three knobs, run "build scale → normalize → shrink → global clip → group int4 quantize → activation-weighted error" for every \((\beta,\text{clip})\) pair, and return the best pair, its scales, its codes and the whole error table.

## Rules

Let \(q_{\max}=2^{\text{bits}-1}-1\). For each \(\beta\) (outer loop) and `clip` (inner loop):

1. **Magnitudes**: \(\text{xm}_j=\frac1T\sum_i|X_{ij}|\), \(\text{wa}_j=\max_o|W_{oj}|\), \(w_{\max}=\max_{o,j}|W_{oj}|\).
2. **Scale**: \(s_j=\text{xm}_j^{\beta}/\text{wa}_j^{1-\beta}\); channels with \(\text{xm}_j=0\) or \(\text{wa}_j=0\) use \(s_j=1\).
3. **Geometric normalization**: \(s\leftarrow s/\sqrt{\max(s)\min(s)}\) (skip when that value is 0), so the scaling does not change the overall magnitude.
4. **Shrink bound**: \(s_j\leftarrow\max(s_j,\,1-\text{max\_shrink})\). With `max_shrink=0` no weight may be shrunk at all.
5. **Global clip**: threshold \(t=\text{clip}\cdot w_{\max}\), \(W^{c}_{oj}=\operatorname{clamp}(W_{oj},-t,t)\). The threshold comes from the **whole matrix**'s max magnitude, not per row — only elements near the global outlier are clipped.
6. **Group quantization**: scale first, \(W^{s}=W^{c}\odot s\) (multiply by input channel), then quantize symmetrically along input channels every `group_size` elements: \(\text{scale}_g=\max|W^{s}_{g}|/q_{\max}\) (scale=1 and all-zero codes for an all-zero group),
   \[
   q_{oj}=\operatorname{clamp}\!\left(\operatorname{round}\!\left(\frac{W^{s}_{oj}}{\text{scale}_{g(j)}}\right),-q_{\max},q_{\max}\right).
   \]
7. **Back to logical weights**: \(\hat W^{\text{log}}_{oj}=q_{oj}\,\text{scale}_{g(j)}/s_j\).
8. **Error** (diagonal approximation of the output error, weighted by channel activation magnitude):
   \[
   \mathcal{L}(\beta,\text{clip})=\frac{1}{O\cdot C}\sum_{o,j}\text{xm}_j\left(W^{c}_{oj}-\hat W^{\text{log}}_{oj}\right)^2.
   \]

Pick the pair with the smallest \(\mathcal L\); ties go to the first occurrence (β index first, then clip index). Everything is computed in float64; rounding follows nearest with ties to even.

Return a five-tuple:

- `best_beta`, `best_clip`: the two optimal floats;
- `best_scales`: float32 \((C,)\), the \(s\) actually used (after normalization and shrink);
- `best_codes`: int64 \((O,C)\), the codes of \(W^c\odot s\) under that pair;
- `losses`: float32 \((\text{len(ratios)},\text{len(clips)})\), `losses[i, j]` for `(ratios[i], clips[j])`.

## Task

- Inputs: `W` float32 \((O,C)\), `X` float32 \((T,C)\), `ratios` (non-empty floats in \([0,1]\)), `clips` (non-empty floats in \((0,1]\)), `max_shrink` (float in \([0,1]\)), `group_size` (positive integer dividing \(C\)), `bits` (2..8).
- Output: the five-tuple above; float tolerance `1e-5` (relative \(10^{-4}\) for the losses).
- Raise `ValueError` when `bits` is out of range; other inputs are guaranteed by the statement.

## Example

```
W = [[ 10.5, -6.0,  2.0],        X = [[0.02, 0.95, 0.48],
     [ -1.8, -2.8,  5.8],             [0.02, 0.06, 0.20],
     [ -2.4, -1.9, -2.6]]             [0.01, 1.36, 1.02],
                                      [0.01, 0.42, 0.94]]
ratios = [0.0, 0.5, 1.0],  clips = [1.0, 0.75, 0.5, 0.25, 0.1]
max_shrink = 0.5,  group_size = 3,  bits = 4
```

`xm = [0.015, 0.6975, 0.66]`, `wa = [10.5, 6.0, 5.8]`, `w_max = 10.5`. Error table (rows = β, columns = clip):

```
              clip=1.0    0.75        0.5        0.25       0.1
β=0.0       1.29940e-2  1.32284e-2  9.88267e-3  3.14876e-3  3.11347e-4
β=0.5       2.00691e-2  2.97825e-2  1.43926e-2  1.80263e-3  1.38912e-4   <- best
β=1.0       3.02391e-2  5.21349e-2  6.84213e-2  2.85086e-2  5.74039e-3
```

```
best_beta   = 0.5
best_clip   = 0.1
best_scales = [0.5, 3.003466, 2.971558]
best_codes  = [[1, -7, 7], [-1, -7, 7], [-1, -7, -7]]
```

(The outlier 10.5 sits in channel 0, whose activations are tiny; clipping it to ±1.05 barely hurts the output yet turns the other columns of its group from "code 0" into full-range codes.)

## Constraints

- The clip threshold is **global** to the layer, `clip * w_max`, not a per-row absmax.
- Normalize with the geometric mean \(\sqrt{\max\cdot\min}\), not the arithmetic mean; apply the `1 - max_shrink` lower bound afterwards.
- Quantization acts on `W_clip * s`; the logical weight in the error must be divided by \(s\) to be comparable with `W_clip`.
- Only the returned values are judged; any implementation style is fine.
