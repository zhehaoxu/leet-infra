# Operator Fusion and Roofline in Depth

## Why an element-wise chain is so wasteful

An element-wise operator does very little compute (a few FLOPs per element) but must read the whole tensor from HBM and write it back. Taking fp16 and
$N=2^{20}$, one read plus one write is 4 MB of traffic while the compute is only about 1 M FLOP — **arithmetic intensity** (FLOP/Byte) is below 1.
The GPU's peak-compute/bandwidth "ridge point" is usually tens to hundreds of FLOP/Byte, so element-wise operators sit far to the left of the ridge and are thoroughly
**memory-bound**: time is almost entirely spent moving data while compute units idle. Chaining $m$ such operators moves the same data back and forth through HBM $2m$ times.

## Fusion collapses the round trips into one

After fusion, intermediates never land in HBM and only flow on-chip; the HBM traffic of the whole chain degenerates into "read in once + write out once" $=2N\,$bytes,
so memory traffic drops to $1/m$ and, under the memory-bound assumption, time also drops by roughly a factor of $m$. This is the core benefit of the **element-wise fusion / horizontal fusion** performed by compilers (torch.compile /
XLA / TensorRT).

## How to read a Roofline

A Roofline plot puts arithmetic intensity AI on the horizontal axis and achievable performance on the vertical axis:
- AI < ridge point (= peak compute / bandwidth) → under the **bandwidth roof**, performance = AI × bandwidth (the region modeled by this problem);
- AI ≥ ridge point → against the **compute ceiling**, where fusion helps little (because the bottleneck becomes compute, e.g. large GEMMs).
This explains why fusion pays off enormously for low-AI operators such as element-wise/reduction kernels and little for large matrix multiplications that are already compute-bound.

## Simplifications of the model vs. reality

- Real fusion is often limited by **on-chip capacity**: if the tile that must fit at once is too large, it has to be split (tiling) and the gain is discounted;
- There is also **kernel launch overhead** (especially for small operators); fusion removes multiple launches, a second benefit besides memory traffic;
- If an operator is a reduction or needs the full data (e.g. global softmax), it cannot be fused purely element-wise and needs two passes or special handling;
- GEMM+epilogue (bias/activation) fusion (persistent kernels / warp specialization) is a focus of modern cuBLASLt,
  CUTLASS and Triton; this problem only accounts for the plainest element-wise bandwidth bill.
