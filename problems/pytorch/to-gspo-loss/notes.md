# 解题说明 · GSPO

## 序列比是 token 比的几何平均

$\prod_t\frac{\pi_\theta(o_t)}{\pi_\text{old}(o_t)}=\frac{\pi_\theta(y)}{\pi_\text{old}(y)}$，所以长度归一后的序列比 $s=\big(\pi_\theta(y)/\pi_\text{old}(y)\big)^{1/|y|}$ **恰好**是逐 token 比值的几何平均。两个直接推论：一是实现上先求和再除长度、最后一次 `exp`，与「逐 token exp 再求几何平均」数值等价但稳定得多；二是任何单个 token 的比值只能把 $s$ 推动 $1/|o|$——长回答里一个 token 的分布漂移不会让整条回答的 ratio 爆炸，这正是 GSPO 想要的平滑性。

## 为什么 token 级裁剪在 MoE 上不稳

token 级 PPO 的代理目标逐 token 裁剪，等价于假设「每个 token 的比值都可以独立调节」。稠密模型里这个近似还行；MoE 模型每个 token 会路由到不同专家，训练一步之后路由可能改变，某个 token 的 $\rho_t$ 可以从 1 跳到几十，触发裁剪、把梯度截断，训练信号变得又噪又偏。序列比把 $T$ 个 token 的漂移平均掉，裁剪决策也只做一次，MoE 上的方差显著更低。这是 GSPO 论文最核心的实验结论。

## 裁剪对象变了，语义也变了

GSPO 裁剪的是**整条回答**的比：$\min(s A,\ \mathrm{clamp}(s)A)$。这带来两个与 token 级不同的行为：同一回答内所有 token 共享「裁/不裁」的决定，不会出现「一半 token 被裁、一半没裁」的碎片化梯度；同时因为 $s$ 是几何平均、通常非常接近 1（$\epsilon$ 取 $3\times10^{-4}$ 的极小值），大多数样本其实根本不触发裁剪，裁剪只在策略整体偏移较大时兜底。

## 与 DAPO/GRPO 的对照

三者都在优化 $\min(\rho A,\text{clip}\cdot A)$，区别在 $\rho$ 的定义域：GRPO/DAPO 用 token 级 $\rho$ 并按 token 平均，GSPO 用序列级 $s$ 并按回答平均。DAPO 的 token 级全局平均强调「每个 token 等权」，GSPO 则彻底回到「每条回答等权」——两者对「长度应该怎样影响权重」给出了相反的答案，放在一起看最能体会这个设计维度。

## 数值细节

`log s` 必须只在有效 token 上求和、除以有效长度；把 padding 的 delta 混进分子或把张量宽度当分母，都会让 $s$ 依赖 padding 量。优势是序列级标量，形状 $(N,)$，不做中心化也不做标准化（那是上游 RLOO/GRPO 的职责），本题只负责"给定优势算损失"这一段。
