# Weight-Only INT8 反量化矩阵乘 详解

## 为什么「只量权重」

推理里显存/带宽瓶颈常在权重：一次前向所有激活才读一遍、而每层权重要被每个 token 反复搬运。weight-only 量化把权重压成 int8/int4（体积 ÷2~4），激活保持 fp16 算——**精度几乎无损**（激活的动态范围大、量化它才掉精度），实现也简单（不用校准激活分布）。这就是 llama.cpp、AutoGPTQ/AWQ 的默认档位 W8A16/W4A16。

## scale 提到矩阵乘外面

$$\text{out}[m,n]=\text{w\_scale}[n]\sum_k x[m,k]\,Wq[n,k]$$

per-channel `w_scale[n]` 与求和变量 $k$ 无关，可以先算 `x @ Wqᵀ`（整数/低精度累加），最后**按输出通道乘一个向量**。两个好处：

1. 不必把整份 int8 权重反量化成 fp16 再存一份（省显存），kernel 里累加器直接乘 scale。
2. 这正是 GPU 上 int8 tensor-core「int32 累加 → per-channel rescale」的数学形式；W4A16 同理，只是解包 nibble。

解法就是这一提法；`F.linear(x, Wq.float()*w_scale[:,None])` 是等价的「先物化反量化权重」参考，两者浮点结果在 `1e-5` 容差内一致（`check_matches_oracle`）。

## 维度与约定

- 权重 $[N,K]$（$N$ 输出、$K$ 输入），`w_scale` 长度 $N$——**按输出通道**，不是输入维，别乘错方向（`[:, None]` vs `[None, :]`）。
- int8 覆盖 −128..127；`Wq.float()` 无损（float32 能精确表示该范围整数）。
- 转置：`x[M,K] @ Wq[N,K]ᵀ -> [M,N]`，与 `F.linear`（内部乘 `Wᵀ`）一致。

## 与量化族其它题的关系

- `per-channel-quantize`：算出这里输入的 `Wq` 与 `w_scale`（fake-quant 那步的真整型版）。
- `group-wise-quantize`：更细的 `w_scale`（每通道多组），matmul 时要按组切换 scale。
- 本合集从「怎么量」（前向 fake-quant）接到「量完怎么算」（推理 matmul）。

## 判题

- oracle：`F.linear(x, Wq.float()*w_scale[:,None])`。
- `_ref` 用三重 `einsum`/显式循环重算，与解法的「矩阵乘 + 列 rescale」不同路径。
- 另有 float64、单输出通道、随机形状用例，均不修改输入。
