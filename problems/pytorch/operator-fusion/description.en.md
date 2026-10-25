# Memory Bandwidth Gains from Operator Fusion (Roofline View)

## Problem

If every operator in an element-wise chain (e.g. `Linear → Bias → ReLU → Dropout`) launches its own kernel, each kernel must **read the intermediate tensor from HBM and write it back to HBM**, so a huge amount of time goes to memory access rather than compute. **Operator fusion** merges adjacent operators into a single kernel, so intermediate results stay on-chip (registers/SRAM) and flow directly, cutting the repeated HBM round trips. This problem quantifies that gain from the **memory-traffic** perspective of the Roofline model.

## Rules

- Implement `solve(n_elems, dtype_bytes, num_ops)`, returning `(non_fused_bytes, fused_bytes, saved_bytes, speedup)`.
- Model: a chain of `num_ops` element-wise operators, where each operator reads exactly **1** and writes **1** tensor of $N$ elements with `dtype_bytes` bytes per element.
- **Non-fused** (one kernel per operator): the $i$-th operator reads its input from HBM and writes its output. $m$ operators amount to $m$ reads + $m$ writes, so the HBM traffic is:
  $$ \text{non\_fused\_bytes} = 2\,m\,N\,\text{dtype\_bytes} $$
- **Fused** (the whole chain in one kernel): the initial input is read once and the final output is written once:
  $$ \text{fused\_bytes} = 2\,N\,\text{dtype\_bytes} $$
- Savings and speedup (assuming kernel time is **HBM-bandwidth bound**, with time proportional to traffic):
  $$ \text{saved\_bytes} = \text{non\_fused} - \text{fused} = 2(m-1)N\,\text{dtype\_bytes} $$
  $$ \text{speedup} = \frac{\text{non\_fused}}{\text{fused}} = m $$
- The first three return integers (bytes); `speedup` returns a float.

## Examples

```
Input: n_elems = 1<<20 = 1,048,576,  dtype_bytes = 2 (fp16),  num_ops = 5
  non_fused_bytes = 2*5*1048576*2 = 20,971,520
  fused_bytes     = 2*1048576*2   =  4,194,304
  saved_bytes     = 16,777,216
  speedup         = 5.0
Output: (20971520, 4194304, 16777216, 5.0)
```

## Constraints

- `n_elems, num_ops` are positive integers; `dtype_bytes` is typically 1/2/4
- This model is a **pure memory roofline**: it assumes each operator reads once and writes once and that time is determined by bandwidth, so the speedup is exactly the operator count $m$; kernel launch overhead, on-chip capacity limits, and compute-bound cases are not counted (see the discussion in notes)
