# conv1d in Depth

## The only difference from conv2d: a single spatial dimension

Reduce conv2d's two sliding dimensions $(H,W)$ to a single length dimension $L$ and you get conv1d. The kernel shape is
$(C_{out}, C_{in}, k_L)$, and the output length is

$$ L_{out} = \left\lfloor \frac{L + 2p - k_L}{s} \right\rfloor + 1 $$

where $p$ is padding and $s$ is stride. Zero padding is applied only on both sides of the length dimension.

## Convolution vs. cross-correlation

Mathematical convolution first flips the kernel by 180°, whereas `conv*` in PyTorch / deep learning actually performs **cross-correlation**
(the kernel is not flipped). The two differ only by a mirrored kernel, which is equivalent for learnable weights (the network will learn the flipped kernel),
so frameworks uniformly use cross-correlation. The reference solution's `x[n,ci,t*s+j] * w[o,ci,j]` is exactly the non-flipped form.

## Common pitfalls in dimension alignment

- **weight channel dimensions**: they are $(C_{out}, C_{in}, k)$, not $(C_{in}, C_{out}, k)$; swapping them yields
  the wrong number of output channels.
- **bias broadcasting**: bias has shape $(C_{out},)$; when added to the output it must be `view(1, -1, 1)` to align with the channel dimension.
- **stride/padding integer division**: the output length uses floor division, consistent with `F.conv1d`'s rounding;
  when $(L+2p-k)$ is not divisible by $s$, it rounds down.
- **Multiple input channels**: each output channel must sum over **all** input channels; don't miss the `ci` loop.

## Applications

1D-CNNs process variable-length/fixed-length sequences: text (WordCNN uses kernels with different $k_L$ in parallel to extract n-gram features),
time-series classification, and waveforms (WaveNet's dilated 1D convolutions). Once you understand conv1d's sliding + channel summation,
conv2d/conv3d just have a few more sliding dimensions, and transposed convolution is the transposed process of "adding to multiple positions".
