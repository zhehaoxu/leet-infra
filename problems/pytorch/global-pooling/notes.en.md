# Global Pooling Explained

## Definition

Reduce over the **entire spatial feature map**, turning $(N,C,H,W)\to(N,C,1,1)$:

- Global Average Pooling (GAP): $\text{out}(n,c)=\frac{1}{HW}\sum_{i,j}x(n,c,i,j)$
- Global Max Pooling (GMP): take the maximum over the whole map

The spatial dimensions are fully absorbed, with no need for flatten + a large fully connected layer.

## Why It Matters

- ResNet uses GAP instead of `Flatten→FC`: the parameter count drops sharply, arbitrary input resolutions are naturally accepted, and overfitting is suppressed.
- CAM (class activation maps) rely precisely on GAP's channel weights × spatial feature map to localize target regions.
- Transformer/ViT also commonly use "global token / mean pooling" to obtain sentence vectors, which is essentially the same idea.

## Implementation Pitfalls

1. **The reduced dimensions must be chosen dynamically by rank**: for 2D it is dim=(2,3), for 3D voxels it is (2,3,4). The general form is `dims=tuple(range(2, x.dim()))`; do not hard-code (2,3).
2. **keepdim=True**: keeping $(N,C,1,1)$ facilitates subsequent broadcasting (e.g. CAM weighting); forgetting keepdim collapses it to $(N,C)$ and breaks the chain.
3. Use `amax` rather than `max` (the latter returns indices and requires `[0]` again).

## Relation to Adaptive Pooling

GAP is a special case of `adaptive_avg_pool2d(x, 1)`; it is also the degenerate case of avgpool with the kernel set to the whole image and an arbitrary stride.
