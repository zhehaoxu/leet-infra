# 量化：KIVI KV Cache 非对称量化（Key 逐通道 / Value 逐 token）

## 题目描述

长上下文推理时 KV cache 比权重还大。**KIVI**（Liu et al., ICML 2024, arXiv:2402.02750）观察到 Key 和 Value 的分布结构不同，于是给两者配了**不同粒度**的非对称量化：

- **Key 逐通道（per-channel）**：Key 的每个特征通道在所有 token 上共享一套量化参数。通道维（`head_dim`）是 Key 分布的「稳定方向」——同一个通道跨 token 的方差大、均值稳定，按通道统计最划算；沿 token 逐个量化反而放大噪声。
- **Value 逐 token（per-token）**：Value 的每个 token 在各个通道上共享一套参数。Value 的分布随 token 剧烈变化，逐 token 量化才能跟上。

两者都用**非对称**（zero-point 不为 0）：权重/激活常整体偏移，对称量化会把一半码字浪费在空区间。本题的量化口径（与 KIVI 一致）：

\[
\Delta = \frac{x_{\max}-x_{\min}}{2^{b}-1}\quad(\text{若 }x_{\max}=x_{\min},\ \Delta=1),\qquad
q=\operatorname{clamp}\!\left(\operatorname{round}\!\left(\frac{x-x_{\min}}{\Delta}\right),0,2^b-1\right),\qquad
\hat x=q\,\Delta+x_{\min}
\]

即 zero-point 就是 `x_min` 本身（浮点，不取整、不 clamp）——`x_min` 精确映射到码 0、`x_max` 映射到码 `2^b-1`。

## 任务

实现 `solve(K, V, bits)`：

- 输入：`K`、`V` 均为 float32 `(T, D)`（token 维在前、通道维在后），`bits` 为 2～8 的整数。
- 对 **K** 沿 token 维（`dim=0`）归约求 min/max：每个通道一套参数，参数长度 `D`；
- 对 **V** 沿通道维（`dim=1`）归约：每个 token 一套参数，参数长度 `T`；
- 输出 `(K_deq, V_deq, K_scale, K_zero, V_scale, V_zero)`：反量化结果 float32 `(T, D)`；`*_scale` 与 `*_zero` 均为 float32，长度分别为 `D`、`D`、`T`、`T`。

**约定**：内部 float64 计算，输出 float32；容差 \(10^{-5}\)。`bits` 非法（`<2` 或 `>8`）抛 `ValueError`。

## 示例

```
K = [[0, 1], [2, 3]], V = [[0, 1], [2, 3]], bits = 4
K 逐通道: ch0 [0,2] → Δ=2/15, min=0 → K_deq[:,0]=[0,2]
          ch1 [1,3] → Δ=2/15, min=1 → K_deq[:,1]=[1,3]
V 逐 token: t0 [0,1] → Δ=1/15, min=0 → V_deq[0]=[0,1]
            t1 [2,3] → Δ=1/15, min=2 → V_deq[1]=[2,3]
```

（每个通道/每个 token 的 min 与 max 都恰好落在码 0 与码 15 上，往返无损。）

## 约束

- 统计方向是本题第一个考点：K 沿 `dim=0`、V 沿 `dim=1`，写反会立刻被粒度断言抓住。
- 退化（`max == min`）时 `Δ = 1`，整组码为 0、重建精确等于原值。
- `round` 为就近取整（银行家舍入，同 `torch.round`）。
