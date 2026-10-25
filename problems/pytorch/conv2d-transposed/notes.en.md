# Transposed Convolution in Depth

## What it actually is

"Deconvolution" is a misleading name — it is not the inverse of convolution, but the **gradient of convolution with respect to the input**.
Let an ordinary convolution be $y = W x$ (a linear operator); then transposed convolution computes $x' = W^\top y$: it **scatter-adds**
each upstream element back onto a larger grid along the kernel. This is also the operation PyTorch's backward pass uses to compute `grad_input`
for a `padding='valid'` convolution.

## Size formula

$$ o = (i - 1)\cdot \text{stride} - 2\cdot \text{padding} + k \;(+ \text{output\_padding}) $$

`output_padding` (not covered in this problem) resolves the ambiguity that, with stride>1, the same input size can correspond to several $o$.
In transposed convolution, `padding` means "how much was cropped when it was treated as an ordinary convolution", so it **reduces** the output — the same direction as in ordinary convolution but with the opposite meaning.

## Implementation notes (the scatter view)

For each input position `(iy,ix)`, its contribution is scattered to output
$(iy\cdot\text{stride}-padding+ky,\; ix\cdot\text{stride}-padding+kx)$,
**discarding out-of-bounds entries and accumulating overlaps**. This is exactly the reverse of ordinary convolution's "gather an input window around each output center", so it is easy to get the direction wrong.

- The weight convention is $(C_{in}, C_{out}, k_h, k_w)$, the opposite of conv2d's $(C_{out}, C_{in},...)$.
- With `stride`, checkerboard artifacts appear in the output, because the contributions of adjacent inputs do not overlap continuously —
  which is why practice often replaces transposed convolution with "nearest-neighbor upsampling + ordinary convolution".

## Applications

Semantic segmentation decoders (FCN/UNet upsampling heads), VAE/GAN generators, super-resolution. UNet also commonly uses
`ConvTranspose2d(stride=2)` for 2× upscaling; checkerboard-sensitive scenarios switch to `upsample+conv`, or use a transposed convolution with `k=stride` so it divides evenly.
