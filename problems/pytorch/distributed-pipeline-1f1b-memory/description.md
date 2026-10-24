# 1F1B 流水线的峰值激活显存

## 题目描述

`distributed-pipeline-bubble` 算的是流水线并行的**时间（气泡）**；本题算它的另一笔账——**激活显存**。
GPipe 为了填满流水线会把所有 micro-batch 的前向先跑完，代价是把 $m$ 份激活全压在每个 stage 的显存里。
**1F1B**（one-forward-one-backward）调度通过交错前向/反向，把每个 stage 同时在飞的 micro-batch 数**封顶**，
大幅削减激活显存。本题算这个峰值（纯计算，自定简化模型）。

## 模型

模型 $L$ 层均分到 $p$ 个 stage，$\text{lps} = L/p$（要求整除）。记 `act_bytes` = **一个 micro-batch 在单层**
需要为反向保存的激活字节数，则一个在飞 micro-batch 在**某 stage**（含其 $\text{lps}$ 层）占
$$ \text{per\_mb} = \text{lps}\cdot \texttt{act\_bytes} $$
一个 stage 的峰值显存 = `per_mb` × 「该 stage 同时在飞的 micro-batch 数」：

- **GPipe**：所有 $m$ 个前向做完才开始反向，每个 stage 都存 $m$ 份
  $$ \text{gpipe\_peak} = \text{lps}\cdot \texttt{act\_bytes}\cdot m $$
- **1F1B**：第 `i` 段（从输入端 $i=0$ 计）在 warmup 期最多同时持有 $\min(m,\; p-i)$ 份在飞激活。于是
  - 最重段 $i=0$：$\text{offfb\_worst} = \text{lps}\cdot \texttt{act\_bytes}\cdot \min(m, p)$
  - 最轻段 $i=p-1$：$\text{offfb\_best} = \text{lps}\cdot \texttt{act\_bytes}\cdot \min(m, 1)$

## 规则

实现 `solve(n_layers, n_stages, n_microbatches, act_bytes)`，返回三元组
`(gpipe_peak, offfb_worst, offfb_best)`（float 字节数）。

## 示例

```
L=32, p=4, m=8, act_bytes=1e6   →  lps=8, per_mb=8e6
gpipe_peak = 8e6·8        = 6.4e7
offfb_worst = 8e6·min(8,4) = 3.2e7      # stage0 少存一半
offfb_best  = 8e6·min(8,1) = 8.0e6      # 末段只存 1 份
→ (6.4e7, 3.2e7, 8.0e6)
```

## 约束

- `1 ≤ n_stages ≤ n_layers`（保证整除才有意义），`1 ≤ n_microbatches`；`act_bytes ≥ 0`
- **`n_layers` 必须能被 `n_stages` 整除**
- 返回 `(float, float, float)`，相对误差 $10^{-6}$ 判等
- 关键：$m\le p$ 时 `offfb_worst == gpipe_peak`（1F1B 不省最重段）；$m>p$ 时才省，最重段比值 $=p/m$
- 只算激活显存，忽略参数/梯度/优化器状态（那些见 `training-state-memory`）
