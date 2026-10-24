# 算子融合与 Roofline 详解

## 为什么逐元素链这么亏

一个逐元素算子的计算量极小（每元素几个 FLOP），但要把整张张量从 HBM 读、再写回。以 fp16、
$N=2^{20}$ 为例，一读一写就是 4 MB 流量，而计算只有约 1 M FLOP——**算术强度**（FLOP/Byte）不到 1。
GPU 的峰值算力/带宽「屋脊点」通常在几十~上百 FLOP/Byte，逐元素算子远在脊点左侧，是彻底的
**memory-bound**：时间几乎全花在搬数据，算力空转。串 $m$ 个这样的算子，就把同一份数据在 HBM
上来回搬 $2m$ 次。

## 融合把往返压成一次

融合后中间结果不落 HBM、只在片上流转，整条链的 HBM 流量退化成「读入一次 + 写出一次」$=2N\,$bytes，
于是访存量降到 $1/m$、在 memory-bound 假设下时间也约降 $m$ 倍。这就是编译器（torch.compile /
XLA / TensorRT）做的 **element-wise fusion / horizontal fusion** 的核心收益。

## Roofline 怎么读

Roofline 以算术强度 AI 为横轴、可达性能为纵轴：
- AI < 脊点（= 峰值算力 / 带宽）→ **带宽屋顶**下沿，性能 = AI × 带宽（本题模型即此区）；
- AI ≥ 脊点 → 贴上**算力天花板**，融合帮助有限（因为瓶颈变成计算，如大 GEMM）。
这解释了为什么融合对逐元素/归约这类低 AI 算子收益巨大，而对已经 compute-bound 的大矩阵乘帮助小。

## 模型的简化与真实差异

- 真实融合常受**片上容量**限制：一次要放下的 tile 过大就得分块（tiling），收益打折；
- 还有 **kernel launch 开销**（尤其小算子），融合省掉多次 launch，是访存之外的第二收益；
- 若某算子是 reduce / 需要全量数据（如全局 softmax），不能纯逐元素融合，需两遍或特殊处理；
- GEMM+epilogue（bias/激活）融合（persistent kernel / warp specialization）是现代 cuBLASLt、
  CUTLASS、Triton 的重点，本题只算最朴素的 element-wise 带宽账。
