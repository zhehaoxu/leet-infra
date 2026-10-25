# Adaptive Max Pooling Explained

## The essence of adaptivity: deriving windows from the output

The output size of a fixed `MaxPool2d(kernel, stride)` is $\lfloor (H-k)/s\rfloor+1$, which changes whenever the input does.
`AdaptiveMaxPool2d(out)` reverses the relationship: given `out` output cells, it slices a segment for each output position using
$$ \text{start}(i)=\lfloor i\cdot H_{in}/H_{out}\rfloor,\quad \text{end}(i)=\lceil (i+1)\cdot H_{in}/H_{out}\rceil $$
When $H_{in}$ is not divisible by $H_{out}$, segment lengths differ by 1 (ceil/floor naturally absorb the remainder), which guarantees that all pixels are covered as evenly as possible and that **every output segment is non-empty**.

## The only difference from adaptive-avgpool

The boundary formulas are identical; only the aggregation changes from `mean` to `max`. Therefore:

- `avg` is smooth and preserves the overall energy level; `max` keeps the **strongest response** and is more sensitive to salient features.
- Combining the two (concat) is a classic attention/descriptor enhancement (e.g. concatenating max+avg pooling before a classification head).

## Max pooling still works for negative numbers

`max` selects the algebraically largest value in the segment, not the largest magnitude: a segment that is entirely negative yields the "least negative" one
(see `check_max_not_mean`). Selecting by magnitude requires `abs` preprocessing, which is a different operator.

## Degenerate cases

- `output_size=1` → the per-channel global maximum, equivalent to global max pooling (many backbones end with
  `AdaptiveMaxPool2d(1)`, which is exactly this).
- Segments may overlap: when $H_{out}>H_{in}$ (upsampling), floor/ceil produce segments of length 1 or even repeated coverage;
  PyTorch also allows this "upsampling-style" adaptive pooling.
