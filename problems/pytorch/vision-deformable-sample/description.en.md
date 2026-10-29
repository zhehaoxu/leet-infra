# Detection: Deformable Sampling and Aggregation

## Problem

Convolution and pooling sample a fixed grid, but many vision tasks need to read values at **continuous positions**: deformable convolution (Dai et al., ICCV 2017) learns an offset per kernel location, and deformable attention (Deformable DETR) takes a set of offset sampling points around each reference point. The offsets are real numbers, so sampling positions land between pixels and only bilinear interpolation applies; positions can also be pushed outside the feature map, where the industry convention is **zero padding**.

This looks like four multiply-adds, but three things go silently wrong in production:

- **Out-of-range corners contribute 0 rather than a replicated edge**: zero padding removes part of the contribution near the border (e.g. at \(x=-0.5\) only the two right corners survive with weight 0.5), whereas clamping duplicates edge pixels and gives different numbers;
- **The offset must actually be added**: the sampling position is `point + offset`, not `point`;
- **Aggregation weights are not normalized**: deformable attention receives weights from an upstream softmax, and the op only performs the weighted sum; dividing by the weight sum inside the op shrinks everything.

The task is a single-scale, batched, optionally weighted deformable sampler. CPU only, deterministic, no RNG.

## Rules

The complete signature, parameter meanings and defaults live only in the starter docstring, which is the sole judging contract.

- `feature` is \( (C,H,W) \) with \(x\) the column index (\(0..W-1\)) and \(y\) the row index (\(0..H-1\)).
- `points` and `offsets` are both \( (N,K,2) \): the former are pixel coordinates (fractional and out-of-range allowed), the latter are offsets \( (\mathrm{d}x,\mathrm{d}y) \), so the sampling position is \( (x+\mathrm{d}x,\ y+\mathrm{d}y) \). The two must share shape and dtype, \(K\ge1\), and \(N\) may be 0.
- Bilinear: for a sampling position let \(x_0=\lfloor x\rfloor\), \(y_0=\lfloor y\rfloor\), \(f_x=x-x_0\), \(f_y=y-y_0\) (floor, including negatives, e.g. \( \lfloor-0.5\rfloor=-1 \)); the four corners \( (x_0,y_0),(x_0+1,y_0),(x_0,y_0+1),(x_0+1,y_0+1) \) carry weights \( (1-f_x)(1-f_y),\ f_x(1-f_y),\ (1-f_x)f_y,\ f_xf_y \). **Every corner outside \( [0,W-1]\times[0,H-1] \) contributes 0** — no clamping, no edge replication, no coordinate truncation; if all four are outside the result is 0.
- Without weights, return \( (N,K,C) \) of sampled values; with \( (N,K) \) weights, return \( (N,C) \) where \( \mathrm{out}[n,c]=\sum_k w[n,k]\cdot\mathrm{sampled}[n,k,c] \), **without dividing by the weight sum** (weights may sum to anything, may be negative, may be all zero).
- The output dtype follows `feature` (internally everything is computed in float64 and cast back), and no input is modified.
- Invalid input raises `ValueError`: `feature` not a tensor, not 3-D, `C`/`H`/`W` zero, or dtype other than float32/float64; `points`/`offsets` not tensors, not \( (N,K,2) \), shape mismatch, dtype different from `feature`, or \(K<1\); `weights` not a tensor, shape not \( (N,K) \), or dtype mismatch; NaN or Inf in `points`/`offsets`/`weights`.

## Example

A \(C=2,\ H=W=3\) feature:

```
ch0 = [[1, 2, 3],      ch1 = [[10, 20, 30],
       [4, 5, 6],             [40, 50, 60],
       [7, 8, 9]]             [70, 80, 90]]
```

With \(N=1,K=3\), three points and offsets:

```
k=0:  point (1,1)   offset ( 0.5, 0  )  -> sample (1.5, 1  )
k=1:  point (0,0)   offset (-0.5, 0  )  -> sample (-0.5, 0 )
k=2:  point (1,1)   offset ( 0.5, 0.5)  -> sample (1.5, 1.5)
```

Expand one by one:

- \(k=0\): \(x_0=1,f_x=0.5,y_0=1,f_y=0\), only `(1,1)` and `(2,1)` carry weight, giving \(0.5\cdot5+0.5\cdot6=5.5\) and \(0.5\cdot50+0.5\cdot60=55\) on the second channel.
- \(k=1\): \(x_0=-1,f_x=0.5,y_0=0,f_y=0\). The corner \((-1,0)\) is outside and contributes 0, the corner \((0,0)\) has weight \(0.5\), so \(0.5\cdot1=0.5\) and \(0.5\cdot10=5\). **Clamping the coordinate to 0 would give 1 instead of 0.5.**
- \(k=2\): all four corners have weight \(0.25\), so \(0.25(5+6+8+9)=7\) and \(0.25(50+60+80+90)=70\).

The sampled result:

```
sampled[0] = [[5.5, 55.0],
              [0.5,  5.0],
              [7.0, 70.0]]
```

Aggregating with weights \( [0.5,0.25,0.25] \):

\[
\mathrm{out}_0=0.5\cdot5.5+0.25\cdot0.5+0.25\cdot7=4.625,\qquad
\mathrm{out}_1=0.5\cdot55+0.25\cdot5+0.25\cdot70=46.25 .
\]

```
aggregated[0] = [4.625, 46.25]
```

Note what the offsets do: without them \(k=1\) would read `1` at \( (0,0) \) and \(k=0\) would read `5` at \( (1,1) \); and weights \( [1,1,0] \) give \(6\) unnormalized versus \(3\) normalized — all of these are discriminators the judge covers.

## Constraints

- Without `weights` return \( (N,K,C) \); with `weights` return \( (N,C) \); dtype matches `feature`.
- The judge uses absolute tolerance \( 10^{-12} \) for float64 input and \( 10^{-5} \) for float32 input; shapes and dtypes are asserted exactly.
- The four-corner/weight order, zeroing of out-of-range corners, offset addition, floor (including negatives), non-normalized aggregation, \(N=0\) and \(K\ge1\) are all part of the contract.
- CPU only, deterministic, no RNG; widely out-of-range coordinates (e.g. \(50,-20\)) must return 0 rather than raising.
