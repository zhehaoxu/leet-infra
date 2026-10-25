# 详解：AdamP 尺度不变方向投影

## 出处
AdamP 论文：*AdamCP / AdamP: Slowing Down the Slowdown for Momentum Optimizers on Scale-invariant Weights*，arXiv:2009.00329（naver，ICLR 2021）。算法骨架对齐 kozistr/pytorch_optimizer 的 `AdamP`（`nesterov=False, cautious=False, foreach=False` 的 per-param 路径，`weight_decouple=True`）。

## 一句话机制
对尺度不变的参数（如归一化权重），沿参数自身方向的更新与衰减都不改变函数值，却会拖慢有效学习并诱发不稳定。AdamP 检测到「梯度与参数近乎正交」时，把更新向量里平行于参数的**径向分量投影掉**，只留切向分量，并把这一组的权重衰减乘以 `wd_ratio`（调小）。

## 关键顺序与陷阱（易错点）
1. **perturb 在投影之前算好**：先由 Adam 矩得到 \(\text{perturb}=\frac{\sqrt{\hat\beta_2}}{\sqrt v+\epsilon}\odot m\)，投影是对这个 perturb 就地减径向。分母 \(\sqrt v+\epsilon\) 用**未偏差修正**的 \(v\)，\(\sqrt{\hat\beta_2}\) 乘在分子（等价于对分母除 \(\sqrt{\hat\beta_2}\)）。
2. **两个视图、命中即 break**：先 channel view（\((C,\cdot)\)，逐通道），再 layer view（\((1,\text{numel})\)）。哪个先满足 `|cos|.max() < delta/sqrt(列数)` 就用它做剔除并 `return`（`r=wd_ratio`），不再检查另一个。列数分别是 `numel/C` 与 `numel`——阈值不同，channel 视图的 `delta/sqrt(列)` 通常更大、更易命中。
3. **归一化用行范数**：\(\hat n = p/(\|p\|_{row}+\epsilon)\)，径向 = \(\hat n\odot\text{perturb}\) 按行求和后 reshape 广播回去，再 `perturb -= \hat n*radial`。注意 \(\hat n\) 的 `+epsilon` 是加在范数上。
4. **只对 dim>1 投影**：一维（偏置、norm 缩放因子）跳过投影，\(r=1.0\)。权重衰减 `p *= 1 − lr*wd*r` 在投影**之后**、参数更新之前施加；`grad is None` 整体跳过。
5. 余弦相似度分母下限用 \(\epsilon\)（`F.cosine_similarity(..., eps=eps)`），再取绝对值。

## 对拍口径
外部参照用独立向量化重算（余弦/投影用与参考解同一份 `_projection`/`_cos_by_view` 帮助函数——因为投影是数据依赖的离散分支，参照与解必须共享同一判据才不会因浮点临界翻转）；Adam 矩部分用显式代数重算。全程 float32，示例值实测 maxdiff 0，随机多步 20 步 ×50 seed ×含 1D/2D/高维混合，最大偏差 ≈2.4e-7，无分支翻转。

## 复杂度
每参数每步 \(O(\text{numel})\) + 逐通道余弦与归一，额外两份状态 \((m, v)\)。
