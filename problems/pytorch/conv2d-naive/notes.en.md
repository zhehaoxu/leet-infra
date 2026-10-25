# Conv2d (Naive Implementation) in Depth

## Definition: it is actually "cross-correlation"

"Convolution" in deep learning does not flip the kernel; it is **cross-correlation**: each output position is the element-wise multiply-add of the receptive field and the kernel:

$$ \text{out}(n,o,i,j)=\sum_{c}\sum_{p,q} x(n,c,\,i\cdot s+p,\,j\cdot s+q)\cdot w(o,c,p,q)+b(o) $$

torch's `F.conv2d` and cuDNN are also cross-correlation, so the naive implementation just needs to match `F.conv2d` — **do not flip the kernel**.

## Shape conventions (the easiest to get backwards)

- Input `x`: $(N, C_{in}, H, W)$
- Weight `weight`: $(C_{out}, C_{in}, k_h, k_w)$ — note that Cout comes first, unlike the input-channel dimension; don't write it backwards.
- Output: $(N, C_{out}, H_{out}, W_{out})$, $H_{out}=\lfloor (H+2p-k_h)/s\rfloor+1$.

## Padding order pitfall

`F.pad`'s pad tuple goes **from the last dimension backwards, (left, right) per dimension**: `(pw,pw,ph,ph)` corresponds to W's left/right and H's top/bottom. Getting the order wrong swaps row and column padding. For convolution, `F.conv2d(..., padding=ph)` is an equivalent shortcut.

## Grouping / dilation (expanding the receptive field)

- `groups=g`: $C_{in}$ and $C_{out}$ are each split into g groups, connected within each group (Depthwise is $g=C_{in}$, $C_{in\_per\_group}=1$).
- `dilation=d`: kernel elements are sampled with gaps, enlarging the receptive field at unchanged parameter count, $H_{out}=\lfloor (H+2p-d(k-1)-1)/s\rfloor+1$.
This problem is the simplest form with $g=1,d=1$; once you understand it, look at the "im2col" and "dilated unfold" problems.

## Naive vs. efficient

The quadruple loop is the teaching version; industrial implementations use im2col+GEMM or direct GEMM/Winograd, turning convolution into matrix multiplication to reuse BLAS throughput (see "im2col" and "batched matrix multiplication").
