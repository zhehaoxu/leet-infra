# Local Response Normalization (LRN)

## Problem Description

Local Response Normalization is a "cross-channel" normalization introduced in the AlexNet era:
for each spatial location it takes a window of size `size` along the **channel dimension** and uses
the sum of squares of the activations in that window to suppress the current channel's response
(numerically similar to lateral inhibition). Early convolutional networks used it to stabilize large
activations and improve generalization; it was later superseded by BatchNorm / LayerNorm, but it
remains an important step for understanding the normalization lineage.

## Rules

- Implement `solve(x, size, alpha=1e-4, beta=0.75, k=1.0)`, where `x` has shape `(N, C, H, W)`.
- For each element, take a window along the channel dimension centered on it with width `size`
  (out-of-range channels are handled by **zero padding**); writing the sum of squares inside the window as $\sum_{j} x_{j}^{2}$, then
  $$ y_{i} = \frac{x_{i}}{\left(k + \dfrac{\alpha}{\text{size}} \sum_{j=\max(0,\,i-\lfloor \text{size}/2\rfloor)}^{\min(C-1,\,i+\lfloor \text{size}/2\rfloor)} x_{j}^{2}\right)^{\beta}} $$
- Normalization slides only along the **channel dimension**, independently for each $(n, h, w)$; $H, W$ do not participate in the window.
- `size` must be odd (so that the window is symmetric about the center). The result must match
  `F.local_response_norm(x, size, alpha=alpha, beta=beta, k=k)`.

## Example

```
Input: x of shape (1,3,1,1), the three channel values = [1, 2, 3]
     size=3, alpha=0.5, beta=1.0, k=1.0
   squared vector = [1, 4, 9], channel window half=1, zero-pad the boundary -> [0,1,4,9,0]
   sum of squares per channel window Σ = [0+1+4, 1+4+9, 4+9+0] = [5, 14, 13]
   denominator = (1 + 0.5/3 · Σ)^1 = [1.8333, 3.3333, 3.1667]
   y = x / denominator = [1/1.8333, 2/3.3333, 3/3.1667]
Output: [0.545455, 0.6, 0.947368]
```

(The window differs per channel: channel 0 only sees [ch0,ch1] plus zero padding on the left, channel 2 only sees [ch1,ch2] plus zero padding on the right,
so the three channels have different denominators.)

## Constraints

- float32 / float64 input; `size` is a positive odd integer; `alpha, beta, k > 0`
- Boundary channels use **zero padding** (equivalent to treating the missing channels' squares as 0); no circular or replicated extension
- Do not use `F.local_response_norm` / `torch.nn.LocalResponseNorm`
