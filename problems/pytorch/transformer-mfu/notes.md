# FLOPs 估算与 MFU 详解

## 6ND：训练算力的经验公式

大模型训练吞吐估算里最著名的是 **$\text{flops} \approx 6ND$**（$N$=参数量、$D$=token 数）：
每个 token 每参数在前向做一次乘加（记 2 FLOPs）→ $2N$；反向的梯度计算量约为前向的 2 倍 → $4N$；
合计 $6N$ 每 token。本题用「每层 matmul 参数量 $P_{\text{layer}}$」把这条经验式拆开写，
更贴近真实结构（注意力/FFN 分开），并额外补上 $QK^\top$、$AV$ 这类**不含参数**的 FLOPs。

## 为什么要单列 attention 的 $4sh$

$2P_{\text{layer}}$ 只覆盖「激活×权重」的投影矩阵乘；而 $QK^\top$ 与 $AV$ 是**激活×激活**，
没有参数参与，容易被漏算。它随序列长度 $s$ **线性**（对每个 query 扫 $s$ 个键）叠加，长上下文时
这部分占比升高——这也是 FlashAttention 等优化的收益来源之一。

## MFU vs HFU

- **MFU**（Model FLOPs Utilization）：只数「模型定义上必须做」的 FLOPs，除以峰值算力。梯度检查点
  的**重计算 FLOPs 不计入分子**，所以开检查点会拉低 MFU（做了更多实际计算换显存）。
- **HFU**（Hardware FLOPs Utilization）：把重计算也算进分子，衡量「硬件被算满」的程度。
两者差额正是重计算开销，工程上常一起看。本题求的是 MFU。

## 判题口径与坑

- 一次乘加 = 2 FLOPs（multiply-accumulate），别按 1 算。
- `step_ms` 是毫秒，换算 FLOP/s 要 `/1000`；achieved 再 `/1e12` 变 TFLOP/s 才能和 `peak_tflops` 对齐。
- embedding / LayerNorm / 激活函数 / dropout 的 FLOPs 在主流估算里都忽略（占比小、且很多是访存bound）。
- MFU 现实里 0.4~0.6 已属优秀（PaLM 报告 GPT 类约 0.5~0.7），别期待接近 1。
