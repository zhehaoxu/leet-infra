# 因果掩码（Causal Mask）详解

## 定义

自回归模型（GPT 类）里，位置 $i$ 只能看 $\le i$ 的 token，不能「偷看未来」。实现是把注意力分数矩阵的**严格上三角**置为 $-\infty$：

$$ \text{mask}_{ij}=\begin{cases}0,& j\le i\\ -\infty,& j>i\end{cases},\qquad \text{scores}+\text{mask} $$

$-\infty$ 经 softmax 后权重变 0，等价于禁止注意未来位置。

## 保留 vs 屏蔽方向（延续 SDPA 语义）

- `keep = torch.tril(ones(bool))`：**下三角 True = 允许**，对角线含在内（位置 $i$ 能看自己）。
- 施加用 `masked_fill(~keep, -inf)`：把上三角（未来）填 $-\infty$。方向反了就变成「只能看未来」。

## 为什么用 -inf 而非大负数

softmax 平移不变，理论上 $-10^9$ 也近似屏蔽，但 $-\infty$ 数学上精确给 0 权重。风险：若某行**全**被 $-\infty$（不存在于标准下三角因果掩码，首行至少保留对角），会 NaN。标准因果掩码每行至少有一个保留位，安全。

## 与其他 mask 组合

padding mask（屏蔽 pad token，见 SDPA 关）与因果 mask 常**同时**用：`keep = causal & pad`。解码器训练时两者叠加。

## 尺寸与设备

掩码形状 $(L,L)$，`device` 要跟随 `scores`，否则跨设备报错。`torch.full`/`ones` 记得传 `device=scores.device`。
