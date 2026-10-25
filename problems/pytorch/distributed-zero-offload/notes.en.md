# Solution Approach

## Interviewer's Intent

The ZeRO family offers three routes to save resources: **partitioning** (saves memory across GPUs, costs cross-GPU bandwidth), **Offload** (dumps optimizer states to the CPU, trading PCIe
for memory, so a single GPU can train large models), and **Infinity** (stacking NVMe on top). Being able to explain exactly what Offload trades for what, and where the bottleneck will be, is the watershed between
"has tuned DeepSpeed configurations" and "understands the memory/bandwidth tradeoff."

## How This Exchange Is Computed

By default, 16 bytes per parameter (fp16 parameters 2 + fp16 gradients 2 + fp32 optimizer 12). After Offload:
- The GPU keeps only the **fp16 parameters** = $2N$ (compared with $16N$ when everything is on the GPU, memory drops to $\frac18$).
- The CPU carries **gradients + optimizer states** = $14N$.
- PCIe traffic per iteration = gradients down $2N$ + updated parameters up $2N$ = $4N$.

Thus $t_{\text{pcie}}=\frac{4N}{\text{BW}}$. With $N=10^9$ and BW$=32$GB/s, $t=0.125$s—this is precisely why a small model
(which computes quickly) actually becomes slower with Offload: $t_{\text{pcie}}$ exceeds the compute time and PCIe becomes the bottleneck. The larger the model and the longer the per-iteration compute,
the easier it is to hide the $0.125$s, and only then does Offload pay off.

## Judging Criteria

`is_bottleneck = t_pcie > compute_time`, **strictly greater than**; equality counts as False—do not write `>=`, `check_bottleneck_strict_gt`
checks exactly this. Any change in bandwidth/parameter count/byte width shows up directly in the four return values; the tests cross-check against an independent reference that accumulates term by term.

## The Triangular Relationship with Its Two Sibling Problems

- `training-state-memory`: the **memory** with everything on the GPU / after ZeRO partitioning ($16N$, $16N/W$).
- `distributed-zero-communication`: the extra **cross-GPU bandwidth** $2cG(+2cP)$ from ZeRO partitioning.
- `distributed-zero-offload` (this problem): Offload replaces "cross-GPU" with "GPU↔CPU PCIe", saving **per-GPU memory**.
Together, the three form the complete memory-bandwidth ledger of the ZeRO family.

## Real-World Engineering Notes

In practice, DeepSpeed overlaps "the forward pass of the next layer" with "the optimizer update + PCIe write-back of the previous layer", so the real $t_{\text{pcie}}$ is often smaller than
the pure bandwidth estimate; ignoring overlap here is a deliberate simplification, and judging only follows the definition in the statement.
