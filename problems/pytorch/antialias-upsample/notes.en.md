# In Depth: Antialiased Resize and Trilinear 3D

## Source

- Operator being replicated: the **CPU** path of `torch.nn.functional.interpolate(x, size, mode="bilinear"|"bicubic"|"trilinear", antialias=..., align_corners=...)` — PyTorch official documentation plus measurements on this repository's torch 2.14.1 CPU build. Officially `antialias` is only allowed for 4-D bilinear/bicubic; the task turns that restriction into part of the error contract.
- **The cubic convolution kernel**: Keys, *Cubic Convolution Interpolation for Digital Image Processing*, IEEE Transactions on Acoustics, Speech and Signal Processing 29(2), 1981 (journal; no arXiv version). Two members of the family are in common use: \(a=-0.75\) (Keys' own recommendation for general interpolation) and \(a=-0.5\) (the Catmull-Rom kernel). Measurements here show native **non-AA bicubic uses \(-0.75\) and the AA path uses \(-0.5\)** — on an upscale configuration the two outputs differ by up to \(0.277\), far beyond any tolerance, so the attribution is reliable. The starter's "PIL convention" label for \(-0.5\) is a third-party-provenance claim we do not re-verify: this repository validates only the **numeric convention**.
- **Why antialiasing when downsampling**: Zhang et al., *Making Convolutional Networks Shift-Invariant Again*, ICML 2019, arXiv:1904.11486 — low-pass (widen the support and normalise by the weight sum) before subsampling, otherwise shrinking aliases and destroys shift invariance. What this task reproduces is how that idea was discretised inside PyTorch's CPU kernel (tap range, the three \(0.5\) offsets, when normalisation happens), not its continuous-domain derivation.
- Versus `upsample-bilinear` (same `operators` collection): that problem is non-AA linear upscale only; here 2-tap/4-tap, AA, align_corners and 3-D are combined into one separable kernel.

## Full derivation and index convention

**Separability**: the resize kernel is a tensor product over the spatial dimensions,
\[\text{out}[o_1,o_2]=\sum_{y}\sum_{x} w_1[o_1,y]\,w_2[o_2,x]\,x[y,x],\]
with each \(w_d\) determined only by \((n_{in},n_{out},\text{align},\text{kernel},\text{AA})\) of that dimension. Applying the per-dimension weights one axis at a time (the reference builds a dense \((n_{out},n_{in})\) matrix and does a matmul) is therefore exactly equivalent to the per-output-pixel double loop used by the judging oracle; `check_channels_batch_independence` additionally asserts "whole batch == per \((b,c)\) slice".

**Half-pixel mapping**: treat a pixel as a unit-width interval; the source interval \([0,n_{in}]\) maps linearly onto the target \([0,n_{out}]\), the centre of target pixel \(dst\) is \((dst+0.5)\cdot\text{scale}\), and expressing it relative to the centre of source pixel 0 gives
\[src=(dst+0.5)\cdot\text{scale}-0.5,\qquad \text{scale}=n_{in}/n_{out}.\]
`align_corners=True` instead pins the first and last **pixel centres**: \(src=\text{scale}\cdot dst\) with \(\text{scale}=(n_{in}-1)/(n_{out}-1)\). Degenerate branch: when \(n_{out}=1\) that ratio has a zero denominator and native uses \(\text{scale}=0\), so \(src\equiv0\) and the whole dimension collapses onto source index 0 (judging asserts `torch.equal(out, x[..., 0:1, :])`; measured native behaves the same).

**Non-AA taps and weights**: the linear kernel uses \(k_0=\lfloor src\rfloor\), \(t=src-k_0\) (clamped to \([0,1]\)) and weights \((1-t,t)\); when \(k_0=n_{in}-1\) the second tap reuses the last index instead of running past it. With half-pixel upscaling the first/last \(src\) is negative (e.g. \(4\to8\): \(src_0=0.5\cdot0.5-0.5=-0.25\)); **the linear kernel clamps \(src\) to 0**, so the first output row gets weights \([1.0,0.0]\) (a copy of source pixel 0). **Bicubic does not clamp \(src\)**: \(k_0=\lfloor-0.25\rfloor=-1\), \(t=0.75\), the four taps \([-2,-1,0,1]\) clamp to \([0,0,0,1]\), with weights \([-0.03515625,\,0.26171875,\,0.87890625,\,-0.10546875]\). After merging equal indices the effective weights are \(w_0=1.10546875,\ w_1=-0.10546875\) (sum still 1) — **this is bicubic's boundary overshoot (ringing)**, and the reason the contract says "clamp tap indices to \([0,n_{in}-1]\) and add weights landing on the same index": overwriting instead of adding leaves \(w_0\) at 0.87890625 and the weight sum at 0.7734, breaking the boundary value and area preservation at once.

**AA window width and normalisation**: let interp_size be the kernel's base support (2 for bilinear, 4 for bicubic).
\[\text{scale}\ge 1\ (\text{shrinking}):\ \text{support}=\tfrac{\text{interp\_size}}{2}\cdot\text{scale},\ \text{invscale}=1/\text{scale};\qquad
\text{scale}<1\ (\text{growing}):\ \text{support}=\tfrac{\text{interp\_size}}{2},\ \text{invscale}=1.\]
The centre is \(\text{center}=\text{scale}(dst+0.5)\) (source-space centre \(=c-0.5\)) and taps run over
\[t\in\big[\max(\lfloor c-s+0.5\rfloor,0),\ \min(\lfloor c+s+0.5\rfloor,n_{in})-1\big],\]
each weight being \(\text{filter}((t-c+0.5)\cdot\text{invscale})\), then divided by the weight sum. Measured tap counts: bilinear \(12\to5\) gives \([4,5,4,5,4]\) and bicubic \(13\to5\) gives \([7,9,11,9,6]\) (caps \(\lceil s\rceil\cdot2+1\) are 7 and 13, never hit). Normalisation is only exact to float32 precision: measured weight sums lie in \([0.9999999,\,1.0000001]\), so "area preservation" is a float32-precision property here, not an identity.

**Why AA equals non-AA for bilinear upscaling**: with \(\text{scale}<1\) we have \(\text{invscale}=1\), \(\text{support}=1\), so taps fall in \([\lfloor c-0.5\rfloor,\ \lfloor c+1.5\rfloor-1]\) — at most the two source indices at \(c-0.5\) and \(c+0.5\) (clamped at the edges) — and the weight \(1-|t-c+0.5|\) is exactly the linear pair \((1-t_\text{lin},t_\text{lin})\). On the boundary row the non-AA path clamps \(src<0\) to \([1.0,0.0]\) while the AA path yields the single tap \([0]\) normalised to \([1.0]\): **both give the same number**, which is the whole equivalence result (measured for \(3\times4\to6\times8\): AA tap counts \([1,2,2,2,2,2,2,2,1]\)). Bicubic keeps support 2 while upscaling but swaps \(a\) from \(-0.75\) to \(-0.5\), so equivalence fails there; and with `align_corners=True` the AA branch still uses the half-pixel centre while scale is \((n_{in}-1)/(n_{out}-1)\), a different phase entirely from non-AA's \(src=\text{scale}\cdot dst\) (measured \(6\to9\) upscale: max difference \(1.23\)).

**Step-by-step float32 rounding**: the reference rounds every intermediate (scale, centre, the \(\lfloor t\pm s+0.5\rfloor\) bounds, each filter value, the normalising division, the running sum) back to float32 via `struct.pack('<f', v)`, imitating the native CPU `float` chain. Measured comparison: computing weights fully in double and casting down afterwards still matches native within \(1.5\times10^{-6}\) (inside the \(10^{-5}\) tolerance), while the float32 chain brings it to \(2.4\times10^{-7}\). So **the rounding chain is not the pass/fail line** — the tap set and the centre convention are; getting a tap wrong pushes the error to order \(10^{-1}\).
**Accumulate in float64**: weights are already frozen at input precision, so the summation runs in float64 and only the final result is cast back; this avoids order-dependent loss when many taps (AA bicubic reaches 11) are summed, and is part of why identity sizes can pass through bitwise (four mode/AA combinations assert `torch.equal`; measured native short-circuits the same-size case too and returns the input bitwise).

**trilinear**: the same linear convention applied independently to \((D,H,W)\); combining it with `antialias=True` raises ValueError (native also refuses).

## Key points and common mistakes

1. **Two Keys parameters**: non-AA bicubic \(a=-0.75\), AA bicubic \(a=-0.5\); hard-coding one makes every bicubic case wrong in one of the two branches.
2. **The AA path ignores the align_corners coordinate convention**: centre is always \(\text{scale}(dst+0.5)\), while scale itself still follows align_corners as \((n_{in}-1)/(n_{out}-1)\).
3. **The non-AA clamp applies only to the linear kernel's \(src\)**: bicubic keeps a negative \(src\) and handles borders by clamping tap indices.
4. **After clamping tap indices, weights on the same index must be added**: bicubic border rows produce repeated indices like \([0,0,0,1]\); overwriting instead of adding silently destroys the kernel.
5. **The AA formula contains three separate \(0.5\) offsets** (\(\lfloor c\pm s+0.5\rfloor\) twice, \((t-c+0.5)\) once); dropping one shifts everything by half a pixel.
6. **Do not widen the window when \(\text{scale}<1\)**: support stays interp_size/2 and invscale stays 1; multiplying by scale unconditionally drags in far-away taps on upscale.
7. **Identity sizes must pass through bitwise**: short-circuit when \(n_{out}=n_{in}\) (or use an identity matrix) so nothing but a weight of exactly 1 participates.
8. **`align_corners=True` with \(n_{out}=1\) has scale 0** — not 1, not NaN — collapsing the dimension onto index 0.
9. **A float32 input should run a float32 computation chain**: computing in double won't fail judging (the tolerance absorbs it), but it breaks any bitwise comparison with native and the exact tap-boundary cases.
10. **Accumulation dtype and final cast**: accumulate in float64, cast back to the input dtype at the end; output dtype must equal input dtype (asserted).
11. **Every error is a ValueError**: AA+trilinear, wrong rank, invalid `size`, non-float dtype — don't copy native's RuntimeError types.

## Differences from neighboring problems

- **vs `upsample-bilinear`**: that problem covers only non-AA linear upscaling; the AA branch, bicubic and 3-D are strict supersets of it here.
- **vs `upsample-nearest`**: nearest takes a single index with \(\lfloor\cdot\rfloor\), has no weights or normalisation, and has no "AA equivalence" question at all.
- **vs `grid-sample`**: there the source coordinates come from an external \([-1,1]\) grid (possibly non-affine); here they are generated from the size ratio. Both clamp at borders, but one clamps to 0 and the other follows `padding_mode`.
- **vs `adaptive-maxpool` / `global-pooling`**: those fix \(o\) and split ranges with floor/ceil, reducing by max/mean; here the reduction is a weighted sum whose weights have negative lobes.
- **vs `conv2d-naive` / `unfold-dilated-conv`**: convolution weights are learned and position-independent (shift equivariant), whereas resize weights are geometric and differ per output position — resize is *not* shift invariant, which is exactly what AA repairs for the shrinking direction.
- **vs `soft-nms` / `topk-select`**: the same "compute per-position weights then merge" shape, but with no coordinate mapping or kernel function involved.

## Cross-check convention and complexity

- **Primary oracle is native `F.interpolate`**: 18 float32 configurations (bilinear/bicubic × both AA states × both align_corners states × upscale/downscale/mixed/identity, plus 4 trilinear groups) and 5 float64 groups; `allclose` at \(10^{-5}\) for float32 and \(10^{-12}\) for float64.
- **Independent oracle: per-output-pixel pure-Python double-precision loops** (`_oracle_bilinear_2d`) that recompute the per-dimension taps/weights themselves and accumulate \(w_y w_x x[y][x]\) in a double loop — a different path from the reference's weight-matrix matmul; six bilinear configurations including AA+align_corners=True (the half-pixel-centre convention) and a \(7\times5\to4\times6\) shrink/upscale mix.
- Discriminating cases: identity-size bitwise pass-through over four mode/AA combinations plus one trilinear case; bilinear upscaling to (6,8)/(7,9) where the AA states must be **bitwise equal** and each also matches its own native state; bicubic upscaling where the two states must differ (asserted `not allclose(atol=1e-4)`); `align_corners=True` with \(n_{out}=1\) collapsing each dimension (asserted against native too); batch/channel independence (bicubic-AA and trilinear); 7 ValueError cases.
- Complexity: separable weight construction costs \(\sum_d O(n_{out,d}\cdot n_{in,d})\), and applying the weights costs \(O(N C \prod_d n_{out,d}\cdot n_{in,d})\); under AA the taps per row are bounded by \(\lceil\text{support}\rceil\cdot2+1\) (up to 11 observed in judging). The reference stores each dimension's weights as a dense \((n_{out},n_{in})\) matrix, trading a little memory for simplicity and float64 accumulation.
