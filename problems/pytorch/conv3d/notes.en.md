# conv3d in Depth

## From conv2d to conv3d: one more sliding dimension

conv3d slides the kernel over three dimensions $(D,H,W)$; the weight shape is $(C_{out}, C_{in}, k_D, k_H, k_W)$,
and the compute has one more kernel dimension $k_D$ than conv2d, making it a major consumer of GPU memory and compute. The output length along each dimension follows the same floor formula
$L_{out}=\lfloor (L+2p-k)/s\rfloor+1$.

## Two hand-written formulations

1. **gather (per output point)**: for each output position, slice out the window and then multiply by the kernel and sum — triple/quintuple loops, intuitive but slow.
2. **scatter (per kernel offset)**: adopted by this reference solution. The outer loop iterates over kernel offsets $(di,hi,wi)$ and multiplies the scalar
   `w[o,ci,di,hi,wi]` onto a **strided slice** of the input `xp[:,ci, di::s, hi::s, wi::s]`,
   accumulating into `out[:,o]`. This turns "the $k^3$ multiply-adds within a window" into "$k^3$ whole-slice vectorized accumulations",
   avoiding a Python loop over every output pixel in the innermost level — a teaching compromise between readability and performance.

Slice bounds: `di : di+od*stride : stride`, ensuring exactly `od` elements are extracted, aligned with the shape of `out[:,o]`.

## Common pitfalls

- **F.pad argument order**: for a 5D tensor, the pad tuple is `(W left, W right, H top, H bottom, D front, D back)`, i.e. **from the last dimension backwards**.
  Filling all six entries with `padding` gives isotropic zero padding across the three dimensions.
- **bias broadcasting**: `view(1,-1,1,1,1)` to align with `(N,Cout,D,H,W)`.
- **Dimension order**: it is `(Cout, Cin, kD, kH, kW)`; don't swap Cin/Cout.

## Applications

Medical voxel segmentation (V-Net, 3D U-Net), video action recognition (C3D; SlowFast uses 3D kernels to extract spatio-temporal features),
point clouds/occupancy grids. 3D convolution is expensive, so engineering practice often uses a (2+1)D decomposition: first a $(1,k_H,k_W)$ spatial kernel, then a $(k_D,1,1)$
temporal kernel, splitting the $k_D k_H k_W$ compute apart — an efficient approximation of conv3d.
