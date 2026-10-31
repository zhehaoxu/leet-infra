# Quantization: vqllm std Normalization + Multi-Step Residual Vector Quantization

## Description

The KV cache stores one vector per token, and the same channel can differ by two orders of magnitude in amplitude across tokens: sharing one static codebook, large-magnitude tokens crowd out all the codewords, while small-magnitude tokens can only land on the nearest "large codeword". vqllm (arXiv 2410.15704, ENLSP-IV 2024) handles this in two steps: first divide each token's vector by its own standard deviation, then perform **residual vector quantization** (RVQ) in the normalized coordinates -- one quantizer is assembled from $N$ sub-codebooks of $\hat d$ dimensions each, and each round takes the nearest codeword for every sub-segment, subtracts it, and hands the remaining residual to the next round; you can run several rounds or move on to the next quantizer. The key side adds one more step, **channel reordering**: the segments of $\hat d$ adjacent channels are transposed and interleaved, so that the same sub-codebook sees coordinates that cross channels instead of one contiguous run of channels.

This problem only tests the deterministic inference part of that chain. Upstream trains the codebooks with `scipy.cluster.vq.kmeans2` and then updates the codewords with an EMA; all of that belongs to the calibration side and is out of scope: the codebooks are given as explicit arguments.

## Rules

Write `x` as $(B,L,D)$; `codebooks` is a list of length $Q$, each entry $(N,M,\hat d)$ with $N\cdot\hat d=D$. Internally everything is float64, and the return values follow the input dtype.

1. **Normalization**: `scale = x.std(dim=-1, keepdim=True)`, i.e. the **unbiased** standard deviation (dividing by $D-1$); positions where `scale == 0` are replaced by 1, then `z = x / scale`. With $D=1$ the unbiased standard deviation is NaN, so $D\ge 2$ is required.
2. **Reordering** (only when `reorder_channel=True`; upstream uses it on keys only): view the last dimension as `view(B,L,N,dhat)`, `transpose(-1,-2)` into $(B,L,\hat d,N)$, then `reshape` back to $(B,L,D)$. Equivalently, collect the channels interleaved by $c_{m\hat d+k}$: with $N=2,\hat d=2$, $[c_0,c_1,c_2,c_3]\to[c_0,c_2,c_1,c_3]$. Quantization runs entirely in the reordered coordinates, and at the end `split(dhat,-1)` + `stack(-1)` + `reshape` restores them.
3. **Per-round nearest neighbour**: for round $s$ of quantizer $q$, cut the current residual into $N$ segments and compare each segment against all $M$ codewords of the $k$-th sub-codebook by Euclidean distance (either the expansion $-\lVert z\rVert^2-\lVert c\rVert^2+2zc^\top$ or the direct squared difference is fine), taking the closest one; on a distance tie take the **smaller entry number**. The selected codewords are stitched back into $(B,L,D)$ and subtracted from the residual. The same quantizer runs `num_residual_steps` rounds in a row, then the next quantizer takes over.
4. **Indices carry an offset**: the entry number picked by segment $k$ gets $k\cdot M$ added, so all indices land in $[0, N\cdot M)$ and do not collide within one quantizer.
5. **Reconstruction**: `quantized = (z - final residual) * scale` -- multiply the scale back first, then restore the channel order with the inverse of rule 2; `error = ||final residual|| * scale`, the reconstruction error of that token (L2, already scaled back).

Return a 5-key dictionary:

- `scale`: $(B,L,1)$, the standard deviation per token (zeros replaced by 1);
- `residual_norm`: $(Q,S+1,B,L)$, the residual norm **before** each round of each quantizer, with the last column holding the value after all $S$ rounds; the first column of quantizer $q>0$ equals the last column of quantizer $q-1$;
- `indices`: $(Q,S,B,L,N)$ int64, the entry number picked per round per segment (with the $k\cdot M$ offset already added);
- `quantized`: $(B,L,D)$, the reconstruction, with the channel order restored;
- `error`: $(B,L)$, the reconstruction error per token.

## Task

- `x`: a 3-D float32 or float64 tensor, $D\ge 2$;
- `codebooks`: a non-empty list/tuple, each entry 3-D, the same dtype as `x`, sharing $(N,\hat d)$, with $N\ge 1$, $M\ge 1$, $N\cdot\hat d=D$;
- `num_residual_steps`: an int $\ge 1$ (a bool does not count); `reorder_channel`: a bool.

Invalid input raises `ValueError`. The arguments must not be modified anywhere.

## Example

$B=L=1$, $D=4$, $N=2$, $\hat d=2$, $M=2$, two residual rounds, `reorder_channel=True`. The reorder interleaves the segments $(1,3)$ and $(2,6)$ into $(1,2)$ and $(3,6)$; after normalization the two segments land on entry 0 and entry 1 of the first codebook (offset $+2$). In the second round the residual flips sign, so both segments switch to entry 1.

```
x                  = [[[1.0, 3.0, 2.0, 6.0]]]
codebooks          = [[[[0.5, 1.0], [-0.05, -0.1]],
                       [[1.5, 3.0], [-0.1, -0.2]]]]
num_residual_steps = 2
reorder_channel    = True

w            = [1.0, 2.0, 3.0, 6.0]                     # after reordering
scale        = [[[2.160246899469287]]]                 # sqrt(14/3)
z            = [[[0.4629100498862757, 0.9258200997725514,
                  1.3887301496588271, 2.7774602993176543]]]
residual_norm = [[[[3.2732683535398857]], [[0.26226555239285204]],
                  [[0.038319636922768574]]]]            # shape (1, 3, 1, 1)
indices      = [[[[[0, 2]]], [[[1, 3]]]]]              # shape (1, 2, 1, 1, 2)
quantized    = [[[0.9721111047611791, 3.0243456592570017,
                  1.9442222095223582, 6.048691318514003]]]
error        = [[0.08277987685119961]]
```

Note that channels 2 and 3 of `quantized` are $(1.9442, 3.0243)$ in the reordered coordinates and only become $(3.0243, 1.9442)$ after the restore: the reconstructed values are placed back by $[c_0,c_2,c_1,c_3]$.

## Constraints

- **Unbiased standard deviation**: `std` defaults to `correction=1`, so the denominator is $D-1$ rather than $D$. Writing `sqrt(mean(z^2))` makes the whole thing too small, and `scale` and `quantized` go wrong together.
- **Zero scale replaced by 1**: a constant token has standard deviation 0; after the swap, $z$ is the raw vector itself, so this step only keeps NaN/Inf away — it does not neutralise a constant token, and the reconstruction is still the codewords selected for those raw magnitudes. Do not switch this to `clamp_min(eps)`: upstream uses the exact `where(std == 0, 1, std)`.
- **Reordering does not change scale**: the standard deviation is invariant under a channel permutation, so `scale` is identical under both `reorder_channel` settings; but the segment contents differ, so `indices`, `residual_norm` and `quantized` all change.
- **Indices carry the segment offset**: `indices` can reach $N\cdot M-1$, not $M-1$; subtract $k\cdot M$ per segment to get back the codebook entry number.
- **Equal distance takes the smaller entry number**: upstream's `argmax` gives no tie order, so this problem pins it to "smallest entry number", and the tests create ties with duplicate codewords.
- **Splice points of the residual trajectory**: `residual_norm[q][0]` is the norm when quantizer $q$ takes over, and `error` equals the last value of the last quantizer times `scale`.
