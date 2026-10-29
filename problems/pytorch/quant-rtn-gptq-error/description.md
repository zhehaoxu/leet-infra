# 量化：RTN 与 GPTQ 输出 MSE 及逐组误差分位数对比

## 题目描述

**GPTQ**（Frantar et al., arXiv:2210.17323）与朴素的 **RTN（Round-To-Nearest）** 都使用相同的 per-tensor 对称量化网格，但 GPTQ 在量化每一列后用 Hessian 逆把误差补偿给后续列。本题要求对同一张权重 \(W\)、同一个输入 \(X\) 分别走 RTN 与 GPTQ，统计：

1. **输出 MSE**
   \[\text{MSE}=\frac{1}{b\cdot d_\text{out}}\|XW^{\mathsf T}-X\hat W^{\mathsf T}\|_F^2.\]
2. **逐组误差分位数**：将列按 `group_size` 分成若干组，对每组计算平均绝对误差
   \[e_g=\frac{1}{d_\text{out}\cdot\text{group_size}}\sum_{(i,j)\in g}|W_{ij}-\hat W_{ij}|,\]
   再返回 \(0.25,0.5,0.75,0.9\) 四个分位数。

量化口径（RTN 与 GPTQ 共用）：

\[\text{scale}=\max|W|,\quad q_{ij}=\operatorname{clamp}\left(\operatorname{round}\left(\frac{W_{ij}}{\text{scale}}\cdot q_\max\right), -q_\max-1, q_\max\right),\quad \hat W_{ij}=\frac{q_{ij}}{q_\max}\cdot\text{scale},\]

其中 \(q_\max=2^{\text{bits}-1}-1\)。

GPTQ 的列更新规则与 `gptq-column-quantize` 一致：

\[H^{-1}=(H+\lambda I)^{-1},\quad W_{:,k+1:C}\mathrel{-}=\frac{\varepsilon_i}{H^{-1}_{ii}}\cdot H^{-1}_{i,k+1:C},\]

其中 \(\varepsilon_i=W_{:,i}-\hat W_{:,i}\)。

## 任务

实现 `solve(W, H, X, bits, damp, group_size)`：

- `W`：float32 \((d_\text{out}, C)\)。
- `H`：float32 \((C, C)\) 对称半正定 Hessian。
- `X`：float32 \((b, C)\)。
- `bits`、`damp`：标量。
- `group_size`：列方向分组大小，整除 \(C\)。
- 返回 `(mse_rtn, mse_gptq, rtn_quantiles, gptq_quantiles)`：
  - `mse_rtn`、`mse_gptq` 为 float32 标量；
  - `rtn_quantiles`、`gptq_quantiles` 为 float32 \(4\) 维张量，依次是 \(0.25,0.5,0.75,0.9\) 分位数。

## 提示

- RTN 直接对每列 round；GPTQ 在 round 后用误差更新未量化列。
- 分位数用 `torch.quantile` 在展平后的 group 统计上计算。
- GPTQ 的输出 MSE 通常不大于 RTN（在测试中会断言严格小于）。
