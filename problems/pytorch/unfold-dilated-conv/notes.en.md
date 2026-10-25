# unfold and Dilated Sampling In Depth

## What unfold really means (the fundamental trap of this problem)

The third argument of `Tensor.unfold(dim, size, step)` is the **window sliding stride**, **not the dilation**!
It cuts the `dim` axis into windows of length `size` that slide by `step` each time, adding a new dimension
that lists the windows.

> A common misconception: `unfold(2, kH, dilation)` is assumed to "sample points at intervals". What you
> actually get is **contiguous windows** (sliding with stride=dilation); when `dilation>1` the windows become
> misaligned with respect to each other, the window span is still `kH`, and you do not get the "insert zeros
> between kernel taps" effect.

## The correct way to get a dilated convolution receptive field

A kernel with dilation $d$ effectively **stretches** the receptive field to a window of span $(k-1)d+1$, then
takes one point every $d$ inside that window. Two-step implementation:

```
u = x.unfold(2, (kH-1)*d + 1, 1).unfold(3, (kW-1)*d + 1, 1)  # large contiguous window, span = dilated receptive field
out = u[..., ::d, ::d]                                        # sample every d inside the window -> true dilation
```

`step=1` guarantees a window starting at every position, and then `[..., ::d, ::d]` performs **strided
sampling** along both kernel axes, yielding $(B,C,oh,ow,kH,kW)$ where each kernel dim picks original pixels
$d$ apart — that is the dilated receptive-field unrolling.

## Sizes

The large window span is $(k-1)d+1$, and the number of output positions is $oh=H-(k-1)d-1+1$; align with the
torch reference according to the convention given in the problem (the cross-check uses a hand-written
reference, so do not call `unfold(...,dilation)` directly).

## Connections

It shares its origin with "im2col" (both unroll the receptive field into kernel dims); the only difference is
that dilation adds one extra strided-sampling step. Once you understand unfold's step semantics, sliding
windows, dilation, and im2col all fall into place at once.
