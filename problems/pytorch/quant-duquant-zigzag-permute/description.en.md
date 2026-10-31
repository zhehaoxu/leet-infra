# Quantization: DuQuant ZigZag Channel Permutation

## Problem

With block-wise quantization, a few outlier channels with huge magnitudes raise the scale of their whole block and drag down every other channel in it. DuQuant (arXiv:2406.01721) applies a "dual transformation" to spread outliers out: a block-wise rotation plus a **ZigZag channel permutation** that distributes large-magnitude channels evenly across blocks. This problem implements only the deterministic permutation step.

Given a weight matrix `W` (`O` rows, `H` columns) and a block size `B`:

1. The key of each column is its maximum absolute weight:

\[
key_c = \max_r |W_{r,c}|
\]

2. Sort all columns by `(key, original index)` in descending, **stable** order;
3. Let `G = H / B` and deal the sorted list into `G` buckets in a snake pattern: first `0,1,...,G-1`, then `G-1,...,0`, then `0,1,...`, until every column is placed;
4. Sort each bucket again by `(key, original index)` in descending stable order;
5. Concatenate buckets `0..G-1` into the permutation `perm`; the permuted weights are `W[:, perm]`.

Adjacent ranks land in different blocks, and the snake direction keeps the total magnitude per block balanced.

## Task

Implement `solve(W, block_size)`:

- `W`: float32 `(O, H)`, non-empty and finite;
- `block_size`: `int` with `1 <= B <= H` and `H % B == 0`;
- Return a `dict`:
  - `perm` int64 `(H,)`: new column `i` comes from original column `perm[i]`;
  - `permuted` float32 `(O, H)`: `W[:, perm]`.

**Conventions**: keys are only used for ordering and are computed in float64; output is float32. Equal keys keep ascending original index (stable sort). Raise `ValueError` when `W` has an illegal dimension/shape/type, contains non-finite values, or `block_size` is not an integer satisfying the conditions above.

## Example

```
W = [[0.5, 3.0, 1.0, 2.0, 0.25, 4.0],
     [0.1, -2.5, 0.5, -1.0, 0.1, -3.5]]
block_size = 2 -> G = 3 blocks

column keys (max |W[:, c]|) = [0.5, 3.0, 1.0, 2.0, 0.25, 4.0]
sorted desc (idx, key)      = [(5, 4.0), (1, 3.0), (3, 2.0),
                               (2, 1.0), (0, 0.5), (4, 0.25)]
zigzag bucket sequence      = 0, 1, 2, 2, 1, 0
buckets = [[5, 4], [1, 0], [3, 2]] -> perm = [5, 4, 1, 0, 3, 2]
permuted = [[4.0, 0.25, 3.0, 0.5, 2.0, 1.0],
            [-3.5, 0.1, -2.5, 0.1, -1.0, 0.5]]
```

## Constraints

- The bucket count is `G = H / B`, so `H` must be divisible by `B`; `B = 1` degenerates to a global stable sort, and `B = H` gives a single bucket sorted internally.
- The snake deal must follow exactly `0,1,...,G-1,G-1,...,0`; a fixed stride or modulo is wrong.
- Re-sorting inside each bucket is part of the algorithm: upstream first deals columns by global magnitude, then keeps every block sorted by magnitude.
- This problem does **only the channel permutation**: DuQuant's block rotation (random QR and `Rot.pkl`) and learnable equivalent transforms are out of scope.
