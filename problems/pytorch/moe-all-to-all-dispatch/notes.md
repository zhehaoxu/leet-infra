# 解题说明 · MoE 专家并行 All-to-All

## 为什么是 all-to-all

TP 里每张卡拿全部 token 的一部分专家；EP 反过来——每卡拿全部专家的一部分 + 一部分 token。路由决策在本地卡完成后，token 必须**跨卡**送到目标专家所在卡；一次通信就叫 all-to-all：每卡向每卡都发。

## send_matrix 的语义

`send_matrix[i, j]` = 从卡 i 送到卡 j 的 (token, slot) 对数量。**不是 token 数**：一个 token 若 top-k 选到 j1/j2 两个不同卡，就在 `send_matrix[i, j1]` 和 `send_matrix[i, j2]` 各计 1。

## combine 权重

原论文与主流实现（GShard/Megatron/VLLM）都要求 `topk_weights` 每行和为 1（**已经在路由阶段 renorm 过**），因此 `combined[t]` 是各专家输出的加权平均。题面约定 `topk_weights` 已经 renorm，`solve` 不再做二次归一化。

## tokens_per_device 用 ceil

`tokens_per_device = ceil(T / P)` 保证前几张卡装满、最后一张可能少装。`src_device(t) = t // tokens_per_device` 是连续分片。题面示例 T=8, P=2 → tpd=4 恰好整除。

## 与 distributed-all-to-all 的关系

`distributed-all-to-all` 是集合通信原语的**通用**版（不规则分片、支持多轮）；本题专用于 MoE：分片规则由 (num_experts, experts_per_device) 与 (T, k) 决定，是 all-to-all 的一个典型工作负载。
