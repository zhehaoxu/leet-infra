# R-FCN Position-Sensitive RoI Pooling Ledger

Detection heads usually pool each RoI once globally, which erases position information. R-FCN
does the opposite: it emits `k × k` groups of position-sensitive score maps first, lets bin
`(i, j)` read only its own channel, and after pooling **equally** averages the `k²` bin scores
into the RoI's class score — translation invariance stays in the convolution, position
sensitivity moves into the channel grouping.

## Interface

```python
def solve(scores, rois, k) -> dict
```

`scores` has shape `(C, k*k, H, W)`: class `c` owns `k*k` channels and channel `i*k + j` serves
only bin `(i, j)`. `rois` is `(N, 4)` of **integer** feature coordinates `[x1, y1, x2, y2]`
(left-closed, right-open). The ledger is:

| Field | Meaning |
| --- | --- |
| `pooled` | `(N, C)`: the final score per RoI per class |
| `bins` | `(N, C, k*k)`: pooled value per bin |
| `bin_sizes` | `(N, k*k)`: how many pixels each bin covers |
| `bounds` | `(N, 4)`: integer bounds after clipping |
| `empty` | `(N, k*k)`: which bins are empty |

## Input / Output

```
Input: scores=[1 class, k=2, 2x2], the four channels are 0/1/2/3
      rois=[[0, 0, 2, 2]]
Output: bins=[[0, 1, 2, 3]], pooled=[[1.5]], bin_sizes=[[1, 1, 1, 1]]
```

Each bin reads one channel, so `(0+1+2+3)/4 = 1.5`.

## Rules

- **Bins are cut by integer division**: the row span is `[y1 + i*h//k, y1 + (i+1)*h//k)`, the
  column span likewise, with `h = max(y2 - y1, 0)` and `w = max(x2 - x1, 0)`. No float rounding.
- **Channels are row-major**: bin `(i, j)` reads channel `i*k + j`; swapping `i` and `j` picks up
  the transposed map.
- **A bin holds an arithmetic mean**: the mean of that channel over the sub-region, not a sum and
  not a max.
- **The `k²` bins vote equally**: the final score averages the `k²` bin values and is **not**
  weighted by bin area. A one-pixel bin and a nine-pixel bin carry the same weight, so the result
  differs from the global mean over the whole RoI.
- **Empty bins are 0.0 and still counted**: when `h < k` the integer cut collapses leading row
  bins into empty spans (a direct consequence of integer division), and their zeros are included
  in the denominator, diluting the score; a zero-area RoI (`x2 <= x1` or `y2 <= y1`) leaves every
  bin empty and scores 0.
- **Clip before bucketing**: coordinates are clamped into `[0, W]` / `[0, H]` and `bounds` records
  the clamped values; an RoI entirely outside the map can degenerate to zero area.
- **Validation**: `k` is a positive integer (`bool` rejected), `scores` must be 4-D with second
  dimension exactly `k*k`, and `rois` must be `(N, 4)` and fully integral; `N = 0` is legal and
  returns an empty ledger.

## Hints

- `pooled` is not an area-weighted average of `bins`: compute the bin means first, then average
  them equally; the two steps cannot be merged into "concatenate pixels and average once".
- An empty bin cannot be pooled directly — `torch.mean` on an empty tensor returns `nan`, so the
  zero has to be written explicitly.
