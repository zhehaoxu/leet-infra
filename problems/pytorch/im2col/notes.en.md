# im2col: Turning Convolution into a Matrix Multiplication

## Core idea

Convolution = one "receptive field · kernel" dot product at every output position. im2col **unrolls the receptive fields of all output positions into column vectors** arranged into a big matrix, so the whole layer degenerates into **one GEMM**:

$$ \underbrace{\text{cols}}_{(N,\;o\!h\cdot o\!w,\;c_{in}k_hk_w)}\times \underbrace{\text{rows}}_{(c_{in}k_hk_w,\;c_{out})}\;\Rightarrow\;(N,o\!h\cdot o\!w,c_{out})$$

This is the principle behind cuDNN/BLAS-accelerated convolution: trade memory for the inefficient quadruple loop and reuse a highly optimized matrix multiply.

## Constructing cols (the key to this problem)

- `x.unfold(2,kh,sh).unfold(3,kw,sw)` → $(N,C,nh,nw,kh,kw)$, where the last two dimensions are the receptive field of each window.
- Permute to $(N,nh,nw,C,kh,kw)$ and reshape to $(N, oh\cdot ow, c_{in}k_hk_w)$: **rows = output positions, columns = flattened receptive fields**. The column order must match the dimension order used to flatten the weights with `weight.reshape(cout, cin*kh*kw)` (C outermost, kh, kw innermost), otherwise the multiplication is wrong.

## Aligning with the weights

`rows = weight.reshape(cout, cin*kh*kw).T`, shape $(c_{in}k_hk_w, c_{out})$. The column dimension of cols and the row dimension of rows are both $c_{in}k_hk_w$; only then is the same-order dot product valid.

## Variants

- **im2row**: the transposed formulation (receptive fields as rows).
- 3D convolution corresponds to volume2col.
- The reverse **col2im**: backpropagation accumulates "the gradients of each column" back into the input, summing at overlapping positions — which is also why the backward pass cannot simply reshape.

## Sizes

Pad with zeros by hand first (`F.pad`, note the last dimension comes first), $oh=\lfloor(h-kh)/sh\rfloor+1$. im2col only handles $g=1,d=1$; grouped convolution must be done block by block, and dilated convolution inserts zeros between kernel elements (see the "dilated unfold" problem).
