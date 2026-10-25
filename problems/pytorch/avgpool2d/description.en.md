# 2D Average Pooling

## Problem

Avg Pooling averages over each window. It has a well-known denominator ambiguity, `count_include_pad`:

$$
\text{out}[i,j] = \frac{\sum_{\text{window}} x}{\text{denominator}},\qquad
\text{denominator} =
\begin{cases}
k_h k_w, & \text{count\_include\_pad=True}\\
\text{number of non-padding pixels in the window}, & \text{count\_include\_pad=False}
\end{cases}
$$

With padding, boundary windows contain fewer valid pixels — the two conventions give different results at the borders.

## Rules

- Implement `solve(x, kernel_size, stride=None, padding=0, count_include_pad=True)`, where `x` has shape `(N, C, H, W)`.
- Pad the input with zeros first, then sum over each window; the divisor follows the table above.
- Calling `F.avg_pool2d` directly is forbidden; using `unfold` to extract windows, summing, and dividing can be fully vectorized.
- **Pitfall**: with `count_include_pad=False` the denominator may differ for each output position (smallest at the corners); using a single global scalar divisor is a common mistake.

## Examples

```
Input: x = 4x4 = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]], kernel=2, padding=0
Output: [[[3.5, 5.5], [11.5, 13.5]]]
```

## Constraints

- float32/float64; the output size uses the floor formula
- The result must match `torch.nn.functional.avg_pool2d` (both branches of `count_include_pad`)
