# Sum Reduction along an Axis Explained

## What "along which axis" is actually squashing

The intuition for `sum(dim=d)` is: **flatten dimension d, keep the rest**. In 2-D, `dim=0` adds vertically along the rows,
giving the sum of each column (length = number of columns); `dim=1` adds horizontally along the columns, giving the sum of each row (length = number of rows).
Higher dimensions work the same way: the result shape is the original shape with dimension d removed.

$$ \text{out}_{\hat d} = \sum_{i} x_{[\hat d \leftarrow i]} $$

## Why keepdim matters

`keepdim=True` keeps the reduced axis with length 1, so the result can still be **broadcast back** against the original tensor. Typical use:
`x - x.mean(-1, keepdim=True)` in softmax/layernorm — without keepdim the dimensions do not line up and broadcasting goes wrong.

## Isomorphic to other reductions

Replacing `+` in the accumulation with `max/min/·` and the initial value with $-\infty/+1$ gives `amax/amin/prod`;
replacing it with a sum followed by division by $n$ gives `mean`. So the triple "one axis + one aggregation function + whether to keepdim"
suffices to describe every reduction operator, a unified mental model for understanding `torch.sum/mean/max/var`.

## A note on numerical stability

Naive accumulation along a large axis accumulates floating-point error; PyTorch internally uses pairwise summation to reduce it. The teaching implementation
may accumulate sequentially, and the comparison uses an `allclose(atol=1e-5)` tolerance rather than exact equality.
