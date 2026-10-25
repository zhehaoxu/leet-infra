# AdaptiveAvgPool2d Explained

## Definition

You do not provide kernel/stride; you provide the **target output size** $H_{out}\times W_{out}$, and the operator decides which part of the input each output position covers. Classic uses: compressing a feature map of arbitrary resolution into a fixed size (e.g. `adaptive(1,1)` before the classifier = GAP, or Inception's multi-scale pooling).

## PyTorch's boundary formulas (the heart of this problem — do not use integer-division segments)

Output position $i$ covers the half-open input interval $[s_i, e_i)$:

$$ s_i=\Big\lfloor\frac{i\cdot H_{in}}{H_{out}}\Big\rfloor,\qquad e_i=\Big\lceil\frac{(i+1)H_{in}}{H_{out}}\Big\rceil $$

- The start uses **floor** and the end uses **ceil**, so the segments **may have different lengths** (some contain 1 pixel, some 2), and segments may overlap or leave gaps. The output is the mean over that segment.
- Common mistake: cutting the input into $H_{out}$ equal-length segments (`H_in//H_out`) — when $H_{in}$ is not divisible by $H_{out}$ this disagrees with torch.

## Size relationship

- When $H_{in}\ge H_{out}$ this is downsampling; when $H_{in}<H_{out}$ it **duplicates/upsamples** (some segments may have length 1 or adjacent outputs may share a segment), which is also allowed.
- `output_size` may be an int (square) or `(oh,ow)`.

## Implementation approach

Precompute each segment's (start,end) with `_bounds` as above, then use a double loop to slice `x[:,:,hs:he,ws:we].mean((2,3))` into the output. Compare against `F.adaptive_avg_pool2d`.

## Relation to global pooling

`adaptive_avg_pool2d(x, 1)` is equivalent to `global average pooling` — both average over the whole map; see the "Global Pooling" problem.
