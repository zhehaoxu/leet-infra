# 解题说明 · 容量分派与 Token Dropping

## capacity_factor 的含义

$C = \lceil \alpha \cdot T \cdot k / E \rceil$。理想均匀分派下每专家约得 $T \cdot k / E$ 个 token，$\alpha=1$ 就是「刚好装得下平均」；$\alpha<1$ 会主动丢一部分；$\alpha>1$ 允许偏斜。

Mixtral/GShard 论文常用 $\alpha \in \{1.0, 1.25\}$。

## position vs probs drop policy

Megatron 提供两种：
- `position`（本实现）：按 token 在原 batch 中的顺序保留前 $C$ 个。语义直观、可复现，Shazeer OLNN 的原始定义。
- `probs`：按 routing_probs 沿 dim=0（token 维）取 top-$C$，保留最自信的 $C$ 个。丢弃的是「不太确定」的分配。

判题按 position。

## 为什么 renorm 前丢弃 / renorm 后丢弃？

本实现先在 $k$ 个选中值之间**renorm**（和为 1），再按位置丢弃。这意味着「丢弃后再求和」的总权重会小于 1。这是标准的 top-k → renorm → drop 顺序，与 Megatron 一致。

若想「丢弃后再 renorm」，则被丢弃 slot 的权重会分摊给保留的 slot，语义不同。**本实现不 renorm** 第二遍。

## dropped 是整数

`int(routing_map.sum() - final_map.sum())`。测试断言用 Python `int`，不要用 0-dim tensor。
