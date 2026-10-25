# AvgPool2d Explained

## Definition

Average over each window. The output size formula is the same as for maxpool:

$$ H_{out}=\Big\lfloor\frac{H+2p-k}{s}\Big\rfloor+1 $$

## Core pitfall: `count_include_pad`

This is the semantic fork unique to AvgPool; it determines the denominator:

- `True` (torch's default): **the denominator is always the window area $k_h k_w$**; the padded zeros count in both the numerator and the denominator → edge outputs are "diluted" and come out smaller.
- `False`: **divide only by the number of valid (non-pad) pixels**; edge windows have a smaller denominator → no down-weighting due to zero padding.

Both are correct, but the numbers differ; align with the judging convention. Implementation trick: run the same unfold-sum over an **all-ones mask** that was 0-padded, giving the valid count per window as the denominator.

## Implementation approach

`F.avg_pool2d(x, kernel_size, stride, padding, count_include_pad)`. By hand, use `unfold(2,kh,sh).unfold(3,kw,sw).sum((4,5))` and then divide by the denominator. The padding here is filled with 0 (opposite to maxpool's $-\infty$, because 0 is neutral for a sum and only the count needs handling).

## Relation to global average pooling

Setting the window to the whole feature map gives GAP (see the "Global Pooling" problem), the key trick used by ResNet and others to replace fully connected layers.
