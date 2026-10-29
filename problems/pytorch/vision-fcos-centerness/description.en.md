# Vision Detection: FCOS Center-ness and Multi-Level Assignment

## Problem

FCOS (Tian et al., *FCOS: Fully Convolutional One-Stage Object Detection*, arXiv:1904.01355) drops anchors and detects directly at every location of a feature map: the distances \((l,t,r,b)\) from a location \((x,y)\) to the four GT sides are the regression target. That forces two design decisions, and they are the whole of this problem:

- **Multi-level assignment**: the same image lives on several levels with different strides, so which location is responsible for which object size? FCOS answers with a *regress range* — each level only owns objects whose \( \max(l,t,r,b) \) falls inside a given interval; locations outside it neither receive regression loss nor count as negatives, they are **ignored**.
- **Center-ness**: per-location classification produces a crowd of "inside the box but far from the centre" low-quality predictions, so FCOS attaches a soft label \( \sqrt{\frac{\min(l,r)}{\max(l,r)}\cdot\frac{\min(t,b)}{\max(t,b)}} \) to every positive location, multiplies it into the classification loss during training and into the confidence at inference, suppressing corner responses.

You must also implement center sampling: only locations inside the GT's centre region (a square of half-side `center_radius` × that level's stride) are positive; other locations inside the box are ignored. When several GTs can claim a location, FCOS picks the **smallest area** — so small objects are not swallowed by big ones.

CPU only, deterministic, no random numbers.

## Rules

The full signature, parameter meanings and defaults live only in the starter docstring, which is the sole judging contract.

- Sampling locations are given in **absolute pixel coordinates**, each tagged with its `levels` index; per-level strides and scale ranges come from `strides` and `regress_ranges`, both of length equal to the number of levels. An out-of-range level index, mismatched lengths, a non-positive stride, an interval without `lo < hi`, or a negative `center_radius` raises `ValueError`.
- For a location \(p=(x,y)\) and a GT \(g=(x_1,y_1,x_2,y_2)\) the four distances (**possibly negative**, meaning outside the box) are
  \[
  l=x-x_1,\quad t=y-y_1,\quad r=x_2-x,\quad b=y_2-y .
  \]
- Three per-GT predicates:
  - `inside(g)`: \(l,t,r,b\ge 0\), i.e. inside the box (**boundary included**);
  - `in_center(g)`: \( |x-\frac{x_1+x_2}{2}|\le \rho \) and \( |y-\frac{y_1+y_2}{2}|\le \rho \), with \( \rho=\text{center\_radius}\times\text{the level's stride} \);
  - `in_range(g)`: \( \mathrm{lo}\le \max(l,t,r,b)<\mathrm{hi} \) using that level's `regress_ranges`, **left-closed, right-open**; the maximum is taken over **that GT's own** four distances.
  - A candidate GT satisfies all three.
- Assignment: with candidates, take the **smallest area** (area \( =\max(x_2-x_1,0)\cdot\max(y_2-y_1,0) \), ties → smallest GT index) and label the location with its index; with no candidate but inside at least one GT → -2 (ignore); otherwise -1 (negative). With \(G=0\) everything is -1.
- The center-ness of a positive location uses the **assigned GT's** distances:
  \[
  c=\sqrt{\frac{\min(l,r)}{\max(l,r)}\cdot\frac{\min(t,b)}{\max(t,b)}},
  \]
  and a factor is 0 when its \(\max\) is 0 (a location exactly on a box edge, or a degenerate GT). Non-positive locations have center-ness 0.
- `reg_targets` holds \( [l,t,r,b] \) of the assigned GT for positives and zeros elsewhere; its dtype follows `gt_boxes` (float32/float64), `labels` is always int64.

## Example

One level, stride 8, regress range \([0,64)\), `center_radius=1.5` (radius \(1.5\times8=12\)), GT \( [2,2,22,22] \) centred at \((12,12)\), and five sampling locations:

```
loc (4,4):   l,t,r,b = 2, 2, 18,18   max=18 ∈[0,64)  in centre  -> positive  c = 2/18      ≈ 0.111111
loc (12,4):  l,t,r,b = 10,2, 10,18   max=18 ∈[0,64)  in centre  -> positive  c = √(1·2/18) ≈ 0.333333
loc (4,12):  l,t,r,b = 2,10, 18,10   max=18 ∈[0,64)  in centre  -> positive  c = √(2/18·1) ≈ 0.333333
loc (12,12): l,t,r,b = 10,10,10,10   max=10 ∈[0,64)  in centre  -> positive  c = 1.0
loc (24,12): x=24 > 22, outside the box                              -> negative -1
```

```
labels      = [0, 0, 0, 0, -1]
centerness  = [0.1111111111, 0.3333333333, 0.3333333333, 1.0, 0.0]
reg_targets = [[2,2,18,18], [10,2,10,18], [2,10,18,10], [10,10,10,10], [0,0,0,0]]
```

Narrow the regress range to \([0,16)\) and the first two locations have \(\max=18\), outside it: they are still inside the box, so they drop from positive to **ignore**, not to negative:

```
labels = [-2, -2, -2, 0, -1]
```

Note that \((24,12)\) is only 2 pixels outside and is already a negative — the `inside` boundary sits exactly on column \(x=22\).

## Constraints

- Return `(labels, centerness, reg_targets)`: `labels` is \((P,)\) int64 with values in \( \{-1,-2\}\cup[0,G) \); `centerness` is \((P,)\) and `reg_targets` is \((P,4)\), both with the dtype of `gt_boxes`.
- Judging uses `torch.equal` on `labels` and absolute tolerance \(10^{-6}\) (float32) / \(10^{-12}\) (float64) on `centerness` and `reg_targets`.
- Boundary conventions are fixed: `inside` is inclusive, `in_center` is inclusive, `in_range` is left-closed/right-open. Degenerate GTs (zero width or height, area 0) still take part in matching and win on smallest area; their center-ness is 0.
- CPU, deterministic, no random numbers; empty GT and empty point sets must return correctly shaped empty/degenerate results rather than raising.
