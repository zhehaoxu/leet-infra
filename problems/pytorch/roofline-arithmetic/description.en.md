# Roofline Arithmetic Intensity and Performance Ceiling

## Problem Description

The Roofline model uses a single metric to decide whether an operator is **compute-bound** or **memory-bound**,
and gives the performance ceiling attainable on that hardware. It is the theoretical tool behind `operator-fusion` (fusion raises arithmetic intensity).

## Rules

- Implement `solve(flops, nbytes, peak_flops, bandwidth)` and return the 4-tuple
  `(arithmetic_intensity, ridge_point, attainable_flops, is_memory_bound)`.
- **Arithmetic Intensity**: how many floating-point operations per byte moved
  $$ AI = \frac{\text{flops}}{\text{nbytes}} \quad [\text{flop/byte}] $$
- **Ridge point**: the critical arithmetic intensity at which the hardware switches from memory-bound to compute-bound
  $$ \text{ridge} = \frac{\text{peak\_flops}}{\text{bandwidth}} $$
- **Attainable performance (Roofline upper envelope)**:
  $$ \text{attainable} = \min\!\left(\text{peak\_flops},\ AI \cdot \text{bandwidth}\right) $$
- **Bottleneck decision**: $AI < \text{ridge}$ is memory-bound (`is_memory_bound=True`, in which case
  $\text{attainable}=AI\cdot\text{bandwidth}<\text{peak}$); otherwise compute-bound
  ($AI \ge \text{ridge}$, $\text{attainable}=\text{peak\_flops}$; equality counts as compute-bound).

## Example

```
Input: flops=1e9, nbytes=1e7, peak_flops=2e14, bandwidth=1e12
  AI      = 1e9 / 1e7        = 100.0 flop/byte
  ridge   = 2e14 / 1e12       = 200.0 flop/byte
  100 < 200 -> memory-bound, attainable = 100 * 1e12 = 1e14
Output: (100.0, 200.0, 100000000000000.0, True)
```

## Constraints

- `flops, nbytes, peak_flops, bandwidth` are all positive
- Return a 4-tuple: the first three are floats, `is_memory_bound` is a bool
- The judge uses relative tolerance `1e-6` for floats; the boundary case `AI == ridge` is treated as compute-bound
