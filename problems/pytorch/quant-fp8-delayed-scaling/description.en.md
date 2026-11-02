# FP8 Delayed Scaling Recurrence Ledger

Production FP8 training almost never computes the scale from the current batch. Instead the
amax of the last few steps is kept in a sliding window, and the scale computed at the
**previous** step is used to cast the **current** step. The delay has two consequences: the
scale always lags the data by one step, and stale amax values in the window keep the scale
from growing.

## Interface

```python
def solve(amax_seq, history_len=3, fp8_max=448.0, margin=0, algo="max", init_scale=1.0) -> dict
```

`amax_seq` is the per-step observed amax (1-D `float64` tensor; `0`, `inf` and `nan` allowed).
The ledger is:

| Field | Meaning |
| --- | --- |
| `scales` | scale actually used for the cast at step t (the one updated at t-1) |
| `next_scales` | scale after the update at step t |
| `amax_used` | reduced amax at step t (after writing into history, before rolling) |
| `histories` | history snapshot per step (`T × history_len`) |
| `kept` | whether the step kept the previous scale because amax was `<= 0` or non-finite |
| `clamped` | whether the computed scale overflowed and was clamped to the FP32 maximum |
| `final_scale` / `final_history` | scale and window after the whole sequence |

## Input / Output

```
Input: amax_seq=[10.0, 20.0], history_len=2, fp8_max=448.0
Output: scales=[1.0, 44.8], next_scales=[44.8, 22.4], amax_used=[10.0, 20.0],
      histories=[[10.0, 0.0], [20.0, 10.0]]
```

The first step has no history and must use `init_scale=1.0`; the second step reduces to `20`
and produces `448/20 = 22.4`, but the cast of that step still uses `44.8` from step one.

## Rules

- **Five ordered actions per step**: 1) write the current amax into `history[0]`; 2) reduce it
  (`max` = window maximum, `most_recent` = slot 0); 3) compute `sf = (fp8_max / amax) / 2^margin`;
  4) update the scale; 5) roll the window left (`history_len == 1` does not roll) and zero slot 0.
- **An unusable amax keeps the previous scale**: when `amax <= 0` (including exactly 0) or the
  amax is non-finite, `scale` is kept as is — it does **not** fall back to `1.0` and does **not**
  blow up to infinity. An overflow that turns amax into `inf` therefore never shrinks the scale,
  which is the most counter-intuitive part of delayed scaling.
- **Overflow clamp**: when a usable amax is extremely small (say `1e-40`), `fp8_max / amax`
  overflows to `inf` in float32 and the scale is clamped to the `float32` maximum
  `3.4028234663852886e38`. The clamp happens *after* the keep-previous-scale fallback, so the
  fallback branch never triggers a clamp.
- **float32 semantics**: amax is narrowed to float32 when written into the window
  (`1234567.89` becomes `1234567.875`) and the scale recurs in float32; the ledger returns
  those values widened to `float64`.
- **`margin`**: `sf` is divided by `2^margin`, with `margin=0` as the TE default.
- **`fp8_max`**: `448` for E4M3 and `57344` for E5M2; this problem only requires a finite
  positive value.
- **Validation**: `history_len` is a positive integer (`bool` rejected), `algo ∈ {max,
  most_recent}`, `margin` is a non-negative integer, and `fp8_max` / `init_scale` are finite
  and positive; an empty sequence is legal and returns an empty ledger.

## Hints

- Do not "fix the lag" by updating the scale before the cast — that is no longer delayed
  scaling, and `scales[0]` must still be `init_scale`.
- The roll direction is easy to invert: rolling left drops the oldest amax off the end and
  frees slot 0 for the next step.
