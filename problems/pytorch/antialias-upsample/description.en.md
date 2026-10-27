# Antialiased Resize and Trilinear 3D

## Problem

`F.interpolate` is the most often called — and most often mis-configured — operator in vision/video pipelines: for the same `size`, `align_corners` and `antialias` each on or off produce visibly different pixels. This task asks you to replicate its **CPU** behaviour point by point, which means composing three orthogonal choices:

1. **coordinate mapping**: half-pixel (\(src=(dst+0.5)\cdot\text{scale}-0.5\)) versus align_corners (\(src=\text{scale}\cdot dst\) with \(\text{scale}=\frac{n_{in}-1}{n_{out}-1}\));
2. **interpolation kernel**: the 2-tap linear kernel, or the 4-tap Keys cubic convolution kernel (bicubic);
3. **antialiasing**: widen the window proportionally to \(1/\text{invscale}\) when downscaling and normalise the weights by their sum (i.e. low-pass before sampling).

The real exam points are three conventions that only show up when you measure the native kernel: **non-AA bicubic uses Keys \(a=-0.75\) (the OpenCV convention) while the AA path uses \(a=-0.5\) (the PIL convention)**; **the AA path keeps using the half-pixel centre even when `align_corners=True`**; and **for float32 input the scale, centre and weights are rounded to float32 step by step, with only the accumulation done in float64**. Each of the three alone is enough to make a solution diverge from native.

CPU, fully deterministic, no RNG calls.

## Rules

- The full signature, parameter meanings and defaults live only in the starter docstring; that is the sole judging contract. `mode` is `"bilinear" | "bicubic" | "trilinear"`; `size` is a positive integer sequence of length 2 (4-D input) or 3 (5-D input); `x` must be float32 or float64.
- **Coordinate mapping** (independent per dimension): with `align_corners=False`, \(\text{scale}=n_{in}/n_{out}\) and \(src=(dst+0.5)\,\text{scale}-0.5\); for the linear kernel \(src<0\) clamps to 0, while **bicubic does not clamp** (a negative centre only changes the weight phase; border taps are handled by clamping the indices). With `align_corners=True`, \(\text{scale}=(n_{in}-1)/(n_{out}-1)\), \(src=\text{scale}\cdot dst\), no clamping, and when \(n_{out}=1\) the scale is 0 — the whole dimension collapses onto source index 0 (measured native behaviour).
- **Non-AA weights**: bilinear/trilinear use \(k_0=\lfloor src\rfloor\), \(t=\text{clamp}(src-k_0,0,1)\) and the 2 taps \((1-t,\ t)\) (if \(k_0=n_{in}-1\) both taps are the last index); bicubic uses taps \(k_0-1..k_0+2\) with the Keys cubic at \(a=-0.75\) (\(|x|<1\): \(((a+2)x-(a+3))x^2+1\); \(1\le|x|<2\): \(((ax-5a)x+8a)x-4a\)); tap indices are clamped to \([0,n_{in}-1]\) and **weights landing on the same index are added**.
- **AA weights** (measured native CPU behaviour): the window widens only when \(\text{scale}\ge 1\) (that dimension shrinks), with \(\text{support}=\frac{\text{interp\_size}}{2}\cdot\text{scale}\) and \(\text{invscale}=1/\text{scale}\) (interp_size: 2 for bilinear, 4 for bicubic); when \(\text{scale}<1\), \(\text{support}=\frac{\text{interp\_size}}{2}\) and \(\text{invscale}=1\). The centre is \(\text{center}=\text{scale}\,(dst+0.5)\) — **the same half-pixel formula even when `align_corners=True`** — taps run over \([\max(\lfloor c-s+0.5\rfloor,0),\ \min(\lfloor c+s+0.5\rfloor,n_{in})-1]\), each weight is \(\text{filter}((tap-c+0.5)\cdot\text{invscale})\) with bicubic using **\(a=-0.5\)**, and the weights are finally **normalised by their sum** (the degenerate zero-sum case emits 0).
- **trilinear** is the linear kernel applied over three spatial dimensions, and combining `antialias=True` with trilinear raises ValueError.
- **Identity sizes**: when \(n_{out}=n_{in}\) for a dimension that dimension passes through bitwise, in both AA states.
- **Numeric convention**: with float32 input, scale/centre/weights are rounded to float32 step by step (matching the native separable/AA kernels), the accumulation runs in float64 and the result is cast back to float32; float64 input stays float64 throughout.
- **Error contract**: bad mode, bad rank (bilinear/bicubic need 4-D, trilinear needs 5-D), wrong `size` length or non-positive entries, non-float dtype, and AA+trilinear all raise **ValueError**.

## Example

\(x\) is (1,1,3,4) holding \(1..12\), `size=(2,2)`, `mode="bilinear"` (defaults `antialias=False, align_corners=False`):

\[x=\begin{bmatrix}1&2&3&4\\5&6&7&8\\9&10&11&12\end{bmatrix}
\;\longrightarrow\;
\mathrm{out}=\begin{bmatrix}2.5&4.5\\8.5&10.5\end{bmatrix}\]

Step by step: \(scale_y=3/2\), \(scale_x=4/2=2\).
- \(dst_y=0\): \(src=0.5\cdot1.5-0.5=0.25\) → rows 0/1 with weights \((0.75,0.25)\); \(dst_y=1\): \(src=1.75\) → rows 1/2 with \((0.25,0.75)\).
- \(dst_x=0\): \(src=0.5\cdot2-0.5=0.5\) → columns 0/1 with \((0.5,0.5)\); \(dst_x=1\): \(src=2.5\) → columns 2/3 with \((0.5,0.5)\).
- \(\mathrm{out}[0][0]=0.75\cdot\frac{1+2}{2}+0.25\cdot\frac{5+6}{2}=0.75\cdot1.5+0.25\cdot5.5=2.5\); \(\mathrm{out}[1][1]=0.25\cdot7.5+0.75\cdot11.5=10.5\).

Every weight (0.5/0.25/0.75) is exactly representable in binary, so the hand computation, the float32 output and the native result agree **bitwise**.

Contrast (measured on the deterministic pattern inputs judging uses): upscaling a 4×4 to (8,8), `mode="bicubic"` differs by up to \(0.277\) between `antialias=False` and `antialias=True` (\(a=-0.75\) vs \(a=-0.5\)); the same-size `mode="bilinear"` upscale to (6,8) is **mathematically identical** in both states (same taps, same weights) and the two native CPU paths differ only by float32 rounding (measured max \(4.8\times10^{-7}\)), while the reference implementation makes them bitwise equal. In addition, for a \(6\times6\to9\times9\) `align_corners=True` upscale, AA versus non-AA differ by \(1.23\) — the AA branch never applies \(\text{scale}\cdot dst\), it stays on the half-pixel centre.

## Constraints

- Return a new tensor of the same dtype as `x` with shape \((N,C,*size)\); do not mutate the input.
- Judging uses native `F.interpolate` as the primary oracle: \(10^{-5}\) (`atol`/`rtol`) for float32 and \(10^{-12}\) for float64; the identity-size case and the align_corners + \(n_{out}=1\) case are asserted bitwise with `torch.equal`.
- The implementation must be **separable** (compute taps and weights per dimension, then apply per dimension) and use no random numbers; the batch and channel dimensions are mere spectators.
