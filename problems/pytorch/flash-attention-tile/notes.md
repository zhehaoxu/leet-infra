# FlashAttention 在线 softmax 详解

## 为什么要分块 + 在线归约

标准注意力要显式存下 $P=\text{softmax}(QK^\top)V$ 的完整 $L_q\times L_k$ 矩阵，显存 $O(L^2)$、
且反复读写 HBM 是瓶颈。FlashAttention 把 Q 分块放进 SRAM，对每块沿 K/V 分块流式累加，
**从不物化完整注意力矩阵**，把显存降到 $O(L)$、并把 IO 变成瓶颈而非算力。

## 在线（streaming）softmax 递推

softmax 分母需要全局最大值，但分块时看不到全部数据。解法是维护三元组并随新块更新：
对当前块 logits $s$，

$$ m_{new}=\max(m_{old}, \max_j s_j),\quad \alpha=e^{m_{old}-m_{new}} $$
$$ l_{new}=\alpha\, l_{old} + \sum_j e^{s_j-m_{new}},\quad \text{acc}_{new}=\alpha\, \text{acc}_{old} + e^{s-m_{new}}V $$

最后 $\text{out}=\text{acc}/l$。$\alpha$ 是关键：遇到更大的 $m$ 时，把**之前**已累加的 $l,\text{acc}$
按新基准重新缩放，等价于把所有块统一到同一个减数 $m$ 下。

## 本关易错点

- **首块 $m_{old}=-\infty$**：$\alpha=\exp(-\infty-m_{new})=0$，正好把初始的零累量保持为零，别写成 NaN。
- **rescale 顺序**：必须先用旧 $m$ 算 $\alpha$ 和 $p=e^{s-m_{new}}$，再更新 $m\leftarrow m_{new}$；顺序错了缩放就漏了。
- **非整除长度**：末尾块不满 `block`，用切片 `min(i0+block, L)` 兜住，别假设整除。
- **$L_k \neq L_q$**：query 块数按 $L_q$、key 循环按 $L_k$，两者独立。
- 大 logits 场景是回归「忘减 $m$」的探针——不减 max 时 `exp(±880)` 直接 inf。

## 与真实实现的差异

真实 FlashAttention 还把反向重算（不存 P）与前向融合成 kernel，并做 warp 级 tiling；
本关只要求前向的数值等价，理解「rescale running max」这一核心即可。因果掩码则让每个 query
块只遍历到对角线附近的 key 块。
