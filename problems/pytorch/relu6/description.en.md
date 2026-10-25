# ReLU6

## Problem

ReLU6 is simply "ReLU with an upper bound": it clips the activation to the interval $[0, 6]$. Unlike unbounded ReLU, it limits the
output range of a single layer, keeping the dynamic range controllable when quantizing to int8, which is why mobile networks such as MobileNet
chose this activation in their original papers specifically for efficient inference.

## Rules

- Implement `solve(x)`, returning $\operatorname{ReLU6}(x) = \min(\max(x, 0), 6)$ with the shape unchanged.
- Input less than 0 → output 0; input greater than 6 → output 6; identity in between.
- The result must match `F.relu6(x)` / `torch.clamp(x, 0, 6)`, preserving the input dtype.

## Examples

```
Input: x = [-2.0, -0.5, 0.0, 0.5, 2.0, 7.0]
Output: [0.0, 0.0, 0.0, 0.5, 2.0, 6.0]
```

## Constraints

- Supports float32 / float64 input; the output dtype matches the input
- Elementwise, shape-preserving
