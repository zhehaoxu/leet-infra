# Concat / Stack / Split in Depth

## The essential difference between the three "merge/split" operations

- **cat (concat)**: concatenates along an **already existing** dimension without increasing the rank. $(2,3)+(4,3)\xrightarrow{\text{dim}=0}(6,3)$. All dimensions other than the concatenation dimension must be identical.
- **stack**: creates a new dimension and then stacks, **increasing the rank**. $(3,)+(3,)\xrightarrow{\text{dim}=0}(2,3)$. All inputs must have exactly the same shape.
- **split / chunk**: the inverse of cat. `split(dim, size)` splits by size, and `chunk(dim, n)` splits by number of pieces (as evenly as possible).

Choosing cat vs. stack incorrectly is the most common bug: use stack when you want to "add a batch dimension", and cat when you want to "extend an existing dimension".

## Hand-written concat for this problem (understanding the memory layout)

```
out_shape[dim] = sum(t.shape[dim] for t in ts)
out = torch.empty(out_shape); off = 0
for t in ts:
    sl[dim] = slice(off, off + t.shape[dim]); out[sl] = t; off += t.shape[dim]
```

Allocate the target shape and write each block into the corresponding slice along the `dim` axis according to an offset — a direct demonstration that cat is simply "laying blocks out one after another along `dim`". For negative `dim`, do `+= ndim` first.

## Pitfalls

1. **Non-concatenated dimensions must match**: with cat `dim=1`, dim=0/2… must be identical, otherwise an error is raised.
2. Tensors in a 1-D list with inconsistent shapes cannot be cat/stacked — they must first be reshaped to align.
3. `cat` is not in-place and copies data; repeatedly accumulating with cat inside a loop is a performance anti-pattern (O(n²) copying). Pre-allocate instead, or collect and cat once.
4. dtype/device must be unified; mixing them raises an error.

## Related

Multi-head attention `Concat(heads)@W_O`, channel concatenation in DenseNet/FPN, and feature fusion in residual connections are all built on cat/stack. Put it together with `gather/scatter` and you have the full toolkit for "reorganizing tensors along dimensions".
