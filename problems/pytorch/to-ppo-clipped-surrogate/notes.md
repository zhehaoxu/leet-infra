# 解题说明 · PPO/GRPO 截断代理目标

## min + clip 如何构成信任域

单独看 `clip(ratio)`：它把比值压进 $[1-\epsilon,1+\epsilon]$。但 PPO 取的是 $\min(\rho A, \hat\rho A)$ 而不是直接 $\hat\rho A$——这保证目标是一个**悲观下界**（pessimistic bound）：
- $A>0$：只有当 $\rho$ 向上越过 $1+\epsilon$ 时截断项更小、生效，抑制"过度奖励好动作"；向下变差时 $\rho A$ 更小、自然被选中，不去掩盖策略变坏。
- $A<0$：只有当 $\rho$ 向下跌破 $1-\epsilon$ 时截断生效，防止"过度打压坏动作"；向上则用未截断项。
因此梯度在"改善方向越界"时被裁成 0，其余方向照常，等效限制了每步策略更新幅度。

## 为什么给 logprob 而非 ratio

数值上 $\log\pi$ 可能极小，直接存 ratio 易下溢；实现（trl / OpenAI baselines）都传 `logprob`，在 loss 里 `exp(logprob - old_logprob)`。本题沿用该约定，测试覆盖 $\rho$ 明显越界的截断生效分支。

## GRPO 的差异仅在上游

GRPO（Group Relative Policy Optimization）去掉 critic，用同一 prompt 采样一组回答、以组内 reward 的标准化 $(r-\mu)/\sigma$ 作为 token 优势 $A$。目标函数与 PPO-clip 完全相同，故本题直接给定 $A$，把"组内标准化"视为外部步骤。真实 GRPO 还常加一项对参考策略的 KL 正则（本题不含，作为后续候选）。
