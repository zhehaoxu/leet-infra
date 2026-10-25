# Multimodal: Swin Window Partition

## Problem

The Swin Transformer (Liu et al., *Swin: Hierarchical Vision Transformer using Shifted Windows*, arXiv:2111.03861) cuts the feature map into non-overlapping \(\text{window}\times\text{window}\) local windows, and attention is computed only inside each window, reducing the complexity from global quadratic to linear. This problem implements **window partition** (and its invertibility check).

## Rules

Implement `solve(x, window)`. Input `x` (float32, shape \((B,H,W,C)\), where \(H,W\) are divisible by `window`). Return shape \((B\cdot \frac{H}{\text{window}}\cdot \frac{W}{\text{window}},\ \text{window},\ \text{window},\ C)\). The window order is row-major by **(batch, window row, window column)**; each window keeps its internal \((\text{row},\text{col},C)\) spatial layout. It is a pure rearrangement; padding is forbidden.

## Examples

`x` (\(1\times4\times4\times2\), values \(0..31\), `x[0,i,j]=[(i*4+j)*2,(i*4+j)*2+1]`), `window=2`:

```
4 windows in total, row-major by (block row, block column); window 0 covers x[0, 0:2, 0:2, :] (2×2×2):
  row 0: [[0,1],[2,3]]   row 1: [[8,9],[10,11]]
out shape = (4, 2, 2, 2)
```

## Constraints

- CPU float32; \(H,W\) are integer multiples of `window`.
- Bijective rearrangement: the original image can be restored with the inverse operation (window merge), which the tests use for verification.
