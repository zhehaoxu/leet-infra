# SPP-Net Pyramid Pooling Ledger

SPP-Net (1406.4729) pools arbitrary-size feature maps into a fixed length
(1+4+16=21). The official implementation does not compute bin boundaries
inside the SPP layer; instead it translates each level into a max-pooling
layer's `kernel / stride / pad`, and the bin boundaries follow the pooling
layer's generic formula. This problem transcribes BVLC/caffe `spp_layer.cpp`
`GetPoolingParam` and `pooling_layer.cpp` `MaxPoolForward` — **the caffe
windowing differs from the paper pseudocode**, which is the core of this
problem.

## Interface

```python
solve(feature, bin_counts)
```

- `feature`: an H×W 2-D list (floats); `bin_counts`: per-side bin counts per
  level (e.g. [1,2,4], not necessarily sorted).
- Constraints: feature is a regular non-empty finite 2-D list; H, W ≥
  max(bin_counts) (entry validation; note this does **not** guarantee no
  empty bins, see below).

## GetPoolingParam (per level k)

```
kernel = ceil(H / k)          # H and W computed separately; kh≠kw for rects
pad    = (kernel·k − H + 1) // 2   # remainder split symmetrically
stride = kernel
```

## MaxPoolForward (window of bin (ph, pw))

```
hs = max(0, ph·kernel − pad)   # start shifted back by pad, may be negative
he = min(hs + kernel, H)       # end uses the unclamped hs+kernel, then clips
```

- **Overlap**: when pad>0, bin(0,·) and bin(1,·) may overlap — the paper form
  `floor(i·H/k)` / `ceil((i+1)·H/k)` does not. At H=5,k=4 the paper gives
  bin1=(1,3) while caffe's bin(1,·) starts at max(0,2−2)=0, same window as
  bin(0,·).
- **Empty bin**: H=6,k=5 → kernel=2, pad=2; bin(4,·) starts at 6 ≥ H → empty
  window, official output `−FLT_MAX` (−inf here). The H≥k constraint does
  **not** rule this out.
- Window max.

## Output

`{"pooled": flatten in (ph,pw) row-major order concatenated level by level,
"shapes": [[k,k]...], "kernels": [[kh,kw]...], "pads": [[padh,padw]...]}`.

## Judging details

- 5×5 + [1,2,4] is the paper's classic example: 21 dims, k=2 level kernel=3
  and pad=1.
- The oracle enumerates each bin explicitly with `max(..., default=−inf)` for
  empty windows; 200 random cases (edges 1–12, 1–3 levels); plus a paper-form
  window comparison asserting the two conventions can differ.

## Boundaries

- `adaptive-maxpool`/`adaptive-avgpool`: single-level adaptive pooling; this
  problem is the **multi-level pyramid + caffe kernel/pad windowing ledger**
  (overlapping windows and the empty-bin quirk appear only in the SPP
  composition).
- AVE / STOCHASTIC pooling out of scope; only the flatten order of the FC
  concatenation is tested.

## Mutation record

12 non-equivalent mutations all caught: ceil→floor, pad→(r)//2, missing
start/end clamp, reversed concat, max→min, paper-form windows, end from
unclamped hs, H/W kernel swap, empty bin as 0, pad fully on the top side,
pad multiplied into ph, window off-by-one. One **equivalent refactoring**
(`min(hs+kernel,H)` rewritten as `min(max(0,...)+kernel, H)`) recorded in
notes.
