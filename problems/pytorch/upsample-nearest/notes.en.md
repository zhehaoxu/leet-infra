# Nearest-Neighbor Upsampling In Depth

## Index mapping: why `i // scale`

When enlarging by a factor of `scale`, output coordinate `i` must point back to an input coordinate.
PyTorch's `nearest` (and the underlying `downsample_nearest`) uses **proportional mapping followed by
flooring**:

$$ \text{src}(i) = \Big\lfloor i \cdot \frac{H_{in}}{H_{out}} \Big\rfloor = \Big\lfloor \frac{i}{\text{scale}} \Big\rfloor $$

For integer enlargement $H_{out}=\text{scale}\cdot H_{in}$, so it is simply `i // scale` — each run of
`scale` consecutive outputs lands on the same input pixel, forming a `scale×scale` replica block.

## Trade-off against bilinear/interpolation

- **No new values**: the output value set is always identical to the input's, which suits label maps /
  semantic segmentation masks / discrete token grids; interpolation would average class indices into
  meaningless intermediate values.
- **Blocky aliasing**: because there is no smoothing, large factors show obvious squares; use
  `bilinear`/`bicubic` when smoothing is needed (see the "bilinear upsampling" and "bilinear grid sampling"
  problems).

## Note that align_corners only affects linear interpolations

`nearest` is unrelated to `align_corners`; only bilinear/bicubic have the corner-alignment ambiguity.

## Efficient implementation

No triple loop is needed; `repeat_interleave` or direct indexing works:
`x.repeat_interleave(sh, dim=2).repeat_interleave(sw, dim=3)`,
or unroll into `torch.gather`. The teaching implementation keeps the explicit loop only to highlight the
essence of "copying".
