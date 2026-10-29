# CISPO 与 K3：停梯度裁剪权重 + KL 估计

## 题目描述

**CISPO**（MiniMax-M1，MiniMax 2025, *MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention*, arXiv:2506.13585）改写了 PPO 裁剪的用法。PPO 的代理目标是 $\min(\rho A,\ \mathrm{clip}(\rho)A)$：比值出现在目标里，被裁剪的 token 会因 `min` 选中常数项而**完全失去梯度**，长序列训练里这会造成大量 token 浪费。CISPO 换了个位置——把裁剪后的比值当成**停梯度的标量权重**，去乘 $\log\pi$：

$$L_\text{CISPO}=-\frac{1}{N_\text{valid}}\sum_{t}\text{mask}\cdot\mathrm{sg}\big(\mathrm{clamp}(\rho_t,1-\epsilon_\text{low},1+\epsilon_\text{high})\big)\,A\,(\log\pi_\theta)_t .$$

这样每个 token 都保留一份**有界**的梯度（权重落在 $[1-\epsilon_\text{low},1+\epsilon_\text{high}]$ 内），无人被丢弃；而 $\mathrm{sg}(\cdot)$ 保证梯度只走 $\nabla\log\pi$ 这一条路，不会出现「比值本身也在被优化」的额外项。

第二个组件是 **K3 KL 估计器**（Schulman 的 k3，见其博客 *Approximating KL Divergence*）：样本来自当前策略 $\pi$ 时，用 $r=\pi_\text{ref}/\pi$ 构造
$$k_3=r-\log r-1\ \ge 0,$$
它无偏估计 $\mathrm{KL}(\pi\,\|\,\pi_\text{ref})$ 且逐样本非负，比 $-\log r$ 方差更低，常被用来监控/惩罚对参考模型的偏离。

本题给定逐 token 对数概率、参考策略对数概率、掩码与序列级优势，算出 CISPO 损失、K3 KL 与裁剪比例。

## 规则

给定 $\text{logp},\text{logp\_old},\text{ref\_logp}$ 均为 $(N,T)$、掩码同形 bool（至少一个 True）、优势 $(N,)$、$\epsilon_\text{low}\in[0,1)$、$\epsilon_\text{high}\ge0$：

1. **重要性比** $\rho_t=\exp(\text{logp}_t-\text{logp\_old}_t)$。
2. **裁剪权重** $w_t=\mathrm{clamp}(\rho_t,1-\epsilon_\text{low},1+\epsilon_\text{high})$，**对它停梯度**（视作常数）。
3. **策略损失** $L=-\dfrac{\sum_t\text{mask}\cdot w_t\,A\,(\log\pi)_t}{N_\text{valid}}$，其中 $(\log\pi)_t$ 就是 `logp`；$A$ 按回答广播到 token。
4. **K3**：$\text{log\_r}=\text{ref\_logp}-\text{logp}$（参考减当前），$k_3=e^{\text{log\_r}}-\text{log\_r}-1$，$\text{kl\_k3}=\dfrac{\sum_t\text{mask}\cdot k_3}{N_\text{valid}}$。为保证 $e^{\text{log\_r}}$ 不溢出，要求 $\max(\text{ref\_logp}-\text{logp})\le20$，否则抛 `ValueError`。
5. **裁剪比例**：$\rho$ 严格落在 $[1-\epsilon_\text{low},1+\epsilon_\text{high}]$ **之外**的有效 token 占比（边界上不算越界）。

返回 `loss`、`kl_k3`、`clip_frac`，均为 0 维 float64。

## 示例

```
logp     = [[ 0.1, -0.2,  0.0], [ 0.3,  0.0,  0.5]]
logp_old = 全 0
ref_logp = [[ 0.0,  0.0,  0.0], [ 0.4,  0.0,  0.0]]
mask     = [[True, True, True], [True, True, False]]
adv      = [1.0, -0.5]        ε_low = ε_high = 0.2

ρ = [[1.105171, 0.818731, 1.0], [1.349859, 1.0, ·]]
w = [[1.105171, 0.818731, 1.0], [1.2,      1.0, ·]]     # 第二行首 token 被上界截到 1.2
loss = -(1.105171·0.1 + 0.818731·(-0.2) + 1.2·0.3·(-0.5)) / 5 = 0.046646
k₃ = exp(ref-logp) - (ref-logp) - 1 = [0.004837, 0.021403, 0, 0.005171]
kl_k3 = 0.031411 / 5 = 0.006282
clip_frac = 1/5 = 0.2      # 只有 ρ=1.349859 越界
```

梯度核对：$\partial L/\partial(\log\pi_t)=-w_t A/N_\text{valid}$，被 mask 的 token 梯度严格为 0；由于 $w$ 已停梯度，未裁剪 token 不会额外多出 $\rho A$ 这一项。

## 约束

- 三个返回值都是 0 维 float64；float32 输入内部升到 float64。
- 停梯度是本题核心：若忘了 `sg`，未裁剪位置的梯度会多出 $\rho A/N$，可用 autograd 对拍发现。
- K3 方向是 `ref - logp`（参考减当前）；逐 token 非负，两者相等时为 0。
- `clip_frac` 用严格不等号统计，$\rho$ 恰在边界时不计入。
- 非法输入抛 `ValueError`：非张量 / 非浮点 / 含 NaN·Inf / 形状不符 / $N$ 或 $T<1$ / mask 非 bool 或形状不符 / 无有效 token / 优势形状或 dtype 不符 / $\epsilon_\text{low}\notin[0,1)$ / $\epsilon_\text{high}<0$ / $\text{ref}-\text{logp}>20$。
- CPU、确定性、不调用随机数。
