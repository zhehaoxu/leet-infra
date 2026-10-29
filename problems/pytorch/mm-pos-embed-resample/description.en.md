# Multimodal: Bicubic Resampling of 2D Position Embeddings

A vision Transformer's position sequence often contains two parts: leading **prefix tokens**, such as CLS, and a two-dimensional spatial grid flattened in row-major order. Changing the spatial resolution should resample only the grid, never mix prefix tokens into spatial neighborhoods.

Separate these parts, explicitly bicubic-resample the old grid to the new grid, then prepend the original prefix tokens in their original order. Both grids may be rectangular. The starter contains the full parameter and return contract.

## Resampling rules

### 1. Protect prefix tokens

Preserve every prefix bit; do not interpolate or change the precision of prefixes. First validate that the old grid matches the spatial token count. If both height and width are unchanged, return a clone of the input. Equal token counts alone do not imply equal grids.

### 2. Half-pixel coordinates

Map each spatial axis independently. Destination index $i$ maps to source coordinate

$$
u=(i+0.5)\frac{n_{\mathrm{src}}}{n_{\mathrm{dst}}}-0.5.
$$

Use the four neighbors $\lfloor u\rfloor+[-1,0,1,2]$. **Do not clamp $u$**. Clamp only the individual lookup indices, computing weights from the original, unclamped neighbor positions. Accumulate every contribution, including repeated boundary indices.

### 3. Keys bicubic kernel

Let $d$ be the absolute distance from the source coordinate to an unclamped neighbor and let $a=-0.75$:

$$
K(d)=
\begin{cases}
(a+2)d^3-(a+3)d^2+1,&0\le d\le1,\\
ad^3-5ad^2+8ad-4a,&1<d<2,\\
0,&d\ge2.
\end{cases}
$$

Multiply the weights from the two axes and sum the $4\times4$ weighted neighbors. Process embedding channels independently, then flatten the spatial grid in row-major order.

Use no antialiasing and no extra offset. Implement the interpolation explicitly; calls to `interpolate`, `grid_sample`, `upsample`, or their aliases/wrappers are forbidden. Native operators are reserved for the testing oracle.

## Example

```text
pos_embed = [[[99.0], [0.0], [8.0]]]
old_hw = (1, 2)
new_hw = (1, 4)
num_prefix_tokens = 1
```

Output:

```text
[[[99.0], [-0.84375], [1.8125], [6.1875], [8.84375]]]
```

This float32 example was recorded from the reference implementation and cross-checked with native non-antialiased bicubic interpolation. The prefix `99.0` is unchanged. Spatial values outside the original range are normal bicubic overshoot; do not clamp them.
