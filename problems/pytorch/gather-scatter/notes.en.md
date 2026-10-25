# Gather / Scatter Explained

## A Pair of Inverse "Move Data by Index" Operations

- **gather**: **read** elements from `input` along `dim` according to `index`.
  $$ \text{out}[i,\dots]=\text{input}[\text{index}[i,\dots]\ \text{along the}\ dim\ \text{axis}] $$
  Rules: `out` and `index` have **the same shape**; every dimension except `dim` must match the corresponding input dimension; only the `dim` axis may differ (values are picked from there).
- **scatter**: **write** `src` into a new tensor according to `index`.
  $$ \text{out}[\text{index}[i]]=\text{src}[i] $$
  one-hot / index_put all rely on it: `zeros.scatter_(dim, index, 1)` builds a one-hot.

## dim Semantics (the most error-prone part here)

`gather(dim=1)` for a 2-D `(row, col)` tensor means "for row `row`, take column `index[row,col]`"; `dim=0` selects across rows. Understand it as "which axis to pick along, with positions on the other axes fixed". For negative `dim`, normalize first with `dim += ndim`.

## Hand-Written Approach (this reference solution): meshgrid advanced indexing

Build an `arange` coordinate grid for each dimension of `index` (`indexing="ij"`), replace the grid layer for `dim` with `index`, then do `input[tuple(gather_index)]`. This exactly reproduces torch `gather` semantics and best illustrates "coordinates on the other dimensions, index on the dim dimension".

## Pitfalls

1. **index must have the same rank and matching shapes on non-dim dimensions**, otherwise broadcasting/shape errors occur.
2. `index` must be of type `long` (`torch.gather` requires integer indices).
3. The `dim_` suffix in scatter is the **in-place** version; with duplicate indices the write order is undefined (use `scatter_add_` to accumulate).
4. For non-contiguous memory/multiple devices in gather, make sure `device` is consistent.
