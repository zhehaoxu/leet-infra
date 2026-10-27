# 量化：NF4 分位码本构造与块编码

## 题目描述

**NF4**（4-bit NormalFloat，Dettmers et al., NeurIPS 2023, arXiv:2305.14314）不是浮点格式，而是一张 16 值的**查表码本**：假设权重服从标准正态，把 \([-1,1]\) 按**等概率**切开，每段用其分位点作重建值。这样在正态权重上逐值重构误差最小（信息论最优），QLoRA 用它把 65B 基座压到 4 bit。

码本的构造（bitsandbytes `create_normal_map(offset=0.9677083, use_extra_value=True)` 的口径）是本题的第一个考点。正半边取 **8** 个概率点、负半边取 **7** 个概率点，再补一个精确的 \(0\)，因此 16 个值关于 0 **不对称**（正侧多一个）：

\[
p_i^{+}=\text{offset}+\frac{0.5-\text{offset}}{8}\,i\ (i=0,\dots,7),\qquad
p_i^{-}=\text{offset}+\frac{0.5-\text{offset}}{7}\,i\ (i=0,\dots,6)
\]

\[
v^{+}_i=\Phi^{-1}(p_i^{+}),\qquad v^{-}_i=-\Phi^{-1}(p_i^{-})
\]

其中 \(\Phi^{-1}(p)=\sqrt2\,\operatorname{erfinv}(2p-1)\)。把 \(\{v^+_i\}\cup\{-v^-_i\}\cup\{0\}\) 共 16 个值排序，再整体除以绝对值最大者归一化到 \([-1,1]\)，即得码本 \(T_0<\dots<T_{15}\)（\(T_0=-1\)、\(T_7=0\)、\(T_{15}=1\)）。

编码沿用 QLoRA 的**块缩放**：每 64 个权重（同一行的连续列）共享一个 absmax 缩放 \(s_{i,\lfloor j/64\rfloor}=\max_j|W_{ij}|\)，块内标量 \(u=W_{ij}/s\)，码取最近码本项

\[
c_{ij}=\arg\min_k |u-T_k|,\qquad \hat W_{ij}=T_{c_{ij}}\,s_{i,\lfloor j/64\rfloor}
\]

并列时取较小的码（与 `argmin` 一致）。块 absmax 为 0 时整块码为 7（\(T_7=0\)）、重建为 0。

## 任务

实现 `solve(w, offset)`：

- 输入：`w`（float32 `(M, N)`，`N` 是 64 的倍数）、`offset`（标量，取值在 \((0.5,1)\)）。
- 输出 `(codebook, codes, deq)`：`codebook` 为 `(16,)` float32；`codes` 为 `(M, N)` int64；`deq` 为 `(M, N)` float32（反量化重建）。

**约定**：内部用 float64 计算，`codebook` 与 `deq` 输出 float32；容差 \(10^{-5}\)。禁止 `bitsandbytes`、`torch.quantile`、`numpy.quantile`。

## 示例

```
w = 形状 (1, 64)，全为 0.5；w[0,0]=1.0，w[0,1]=-1.0，w[0,2]=0.95，offset=0.9677083
absmax=1 → u = w 本身
codes[0,:4] = [15, 0, 15, 12]      # 1.0 与 0.95 都最近 1.0（码 15），0.5 最近 0.4407（码 12）
deq[0,:4]   = [1.0, -1.0, 1.0, 0.4407098]
```

## 约束

- `0.5 < offset < 1`，否则抛 `ValueError`；`N % 64 == 0`。
- 码本 16 个值严格递增、关于 0 不要求对称（正侧 8 个、负侧 7 个）。
- 同一块内权重整体乘 2 时，码不变、重建值乘 2（块缩放线性）。
