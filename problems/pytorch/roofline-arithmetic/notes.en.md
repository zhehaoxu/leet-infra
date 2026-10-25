# Solution Approach

## Interviewer Intent

Roofline is the first-principles tool of performance analysis: given an operator's compute amount and memory traffic, can you immediately tell whether it is stuck on compute or bandwidth, and how much optimization headroom exists? This is a basic skill of kernel / inference optimization, and it directly explains why you should do operator fusion in
`operator-fusion` — the essence is **raising arithmetic intensity**.

## The Three Quantities

1. **Arithmetic intensity $AI=\text{flops}/\text{bytes}$**: a property of the operator itself, measuring "how many operations are bought back per byte moved".
   Elementwise operators have very low $AI$ (read a lot, do one add/multiply), while large matrix products have high $AI$.
2. **Ridge point $\text{ridge}=\text{peak}/\text{bandwidth}$**: a property of the hardware, measuring "how many
   flops per byte this machine needs to keep the compute fed". For example, $200$ TFLOP + $1$ TB/s → you need $AI\ge200$ to avoid wasting compute.
3. **Attainable performance $=\min(\text{peak}, AI\cdot\text{bandwidth})$**: the envelope of the two rooflines (the horizontal compute ceiling + the sloped bandwidth ramp),
   taking the smaller one.

## Decision Rules

- $AI < \text{ridge}$: still on the ramp → **memory-bound**, performance $=AI\cdot\text{bandwidth}$; adding compute is useless here,
  you must **reduce memory traffic** (fuse, increase data reuse).
- $AI \ge \text{ridge}$: already at the plateau → **compute-bound**, performance $=\text{peak}$; now you should optimize the algorithm / use better kernels.
- At the boundary $AI=\text{ridge}$ the two lines intersect, and either side gives the same result (this problem counts it as compute-bound).

## Intuitive Examples

The ridge of A100/H100 is often several hundred flop/byte. So low-$AI$ operators like softmax, layernorm, and elementwise adds are almost always
memory-bound — this explains "why fusing them into the matmul epilogue earns free performance".

## Common Mistakes

- Writing $AI$ as $\text{bytes}/\text{flops}$ (numerator and denominator swapped).
- Answering performance $=\text{peak}$ when memory-bound (in fact the ceiling has not been reached).
- Reversing the comparison direction between ridge and $AI$, making the bottleneck decision inconsistent with the attainable performance.

## Grading Details

`test_cases.py` recomputes the attainable performance independently via the envelope formula `min(peak, AI*bw)` (not reusing solve's if/else), and unit-tests
both the memory-bound and compute-bound sides, the classification of the $AI=\text{ridge}$ boundary, and the invariant "attainable performance is always $\le$ peak".
