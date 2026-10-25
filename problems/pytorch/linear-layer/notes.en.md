# Fully Connected Linear Layer in Depth

## Weight orientation: the historical baggage of `(out, in)`

`nn.Linear` stores the weights as `(out_features, in_features)`, so the forward pass is $y = xW^\top + b$
(the dot product is taken over the last dimension of `x`). Many people first write `x @ weight` and hit a shape mismatch—the correct form is
`x @ weight.T`. This convention comes from column-wise organization in early MATLAB/caffe, and it cannot be changed because a huge number of
`state_dict`s are saved in this order.

## Why it is transparent to batch dimensions

A linear transform acts only on the **last dimension**; all preceding dimensions are treated as independent "samples" processed in parallel:
`(N, L, in) → (N, L, out)`. Implementation-wise this is equivalent to flattening the leading dimensions into rows, doing one matrix multiply, and reshaping back;
PyTorch's broadcasting makes the manual reshape unnecessary.

## Relation to other operators

- **Convolution** can be seen as "a linear layer over a local window": `Conv2d` applies a set of `(out, in·k·k)`
  linear weights to each receptive field (see the im2col problem, which turns this explicitly into a matrix multiply).
- **The QKV projections, FFN, and LM head of attention** are all linear layers, only with different `in/out` sizes.
- **Bias fusion**: inference engines often fold the bias into the matmul epilogue, saving one element-wise pass.

## Gradient perspective

With respect to `x`: $\partial y/\partial x = W^\top$ (mapping the gradient back into input space); with respect to `weight`:
$\partial y/\partial W = \delta^\top x$ (an outer product). Once you understand the dot-product structure of the forward pass, the backward pass is just its transposed rearrangement.
