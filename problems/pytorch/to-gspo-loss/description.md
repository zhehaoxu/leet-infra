# GSPO：序列级重要性比

## 题目描述

**GSPO**（Zheng et al. 2025, *Group Sequence Policy Optimization*, arXiv:2507.18071）重新定义了大模型 RL 里的重要性比：PPO/GRPO 用的是**逐 token** 的 $\rho_t=\pi_\theta(o_t)/\pi_\text{old}(o_t)$，每个 token 单独裁剪；GSPO 把整条回答当成一个动作，用**长度归一化的序列比**

$$s_i=\Big(\frac{\pi_\theta(y_i|x)}{\pi_\text{old}(y_i|x)}\Big)^{1/|y_i|}=\exp\!\Big(\frac{1}{|o_i|}\sum_t\log\frac{\pi_\theta(o_{i,t}|x,o_{i,<t})}{\pi_\text{old}(o_{i,t}|x,o_{i,<t})}\Big)$$

替代它，并让整条回答共用一个裁剪决策。

这么做有两层动机。其一是**方差**：token 级比值的方差随序列长度累积，少数 token 的极端比值会让梯度剧烈波动；序列比是 token 比值的几何平均，单个 token 只能把 $s$ 推动 $1/|o|$，天然平滑。其二是 **MoE 稳定性**：token 级 PPO 的推导隐含「路由在训练步之间一致」，而 MoE 的路由会漂移，导致 token 级比值大范围抖动；序列级比值把这种抖动平均掉，使 MoE 上的 RL 训练稳定得多。

本题给定逐 token 对数概率、掩码与序列级优势，算出 GSPO 损失与序列比。

## 规则

给定 $\text{logp}/\text{logp\_old}$ 形状 $(N,T)$、掩码同形 bool（每条回答至少一个 True）、序列级优势 $A$ 形状 $(N,)$、裁剪半径 $\epsilon\in(0,1)$（原论文取 $3\times10^{-4}$）：

1. **先平均再取指数**：$\log s_i=\dfrac{\sum_t\text{mask}\cdot(\text{logp}-\text{logp\_old})}{|o_i|}$，$s_i=e^{\log s_i}$。不是「逐 token 取指数后平均」，也不是「总和直接取指数」。
2. **整条回答共用一个裁剪比**：$\bar s_i=\mathrm{clamp}(s_i,1-\epsilon,1+\epsilon)$，目标 $\text{obj}_i=\min(s_i A_i,\ \bar s_i A_i)$。
3. **按回答平均**：$L=-\dfrac{1}{N}\sum_i\text{obj}_i$，不再按 token 平均。

返回 `loss`（0 维 float64）与 `ratio`（$(N,)$ float64，就是 $s$）。

## 示例

```
logp = [[ 0.10, -0.20,  0.00],
        [ 0.30,  0.00,  0.50]]      logp_old = 全 0
mask = [[True, True, True],
        [True, True, False]]        adv = [1.0, -1.0]
ε = 3e-4

第 0 条：log s = (0.1 - 0.2 + 0.0)/3 = -0.033333   s = 0.967216
第 1 条：log s = (0.3 + 0.0)/2        =  0.150000   s = 1.161834
obj₀ = min(0.967216×1, 0.9997×1) = 0.967216
obj₁ = min(1.161834×(-1), 1.0003×(-1)) = -1.161834
loss = -(0.967216 - 1.161834)/2 = 0.097309
```

第 0 条 $s<1$、优势为正，`min` 取未裁剪项；第 1 条优势为负，`min` 同样取更负的未裁剪项——负优势侧的裁剪上界并不生效，这是 PPO 式代理目标的固有行为。

## 约束

- 返回 `loss` 0 维、`ratio` $(N,)$，均为 float64；float32 输入内部升到 float64。
- 掩码位置不进入对数比：把被 mask 的 logp 改成任意值，$s$ 与损失都不变。
- 长度归一在**指数之前**：同样的 delta 总和，摊到 3 个 token 与 1 个 token 得到不同的 $s$。
- 按回答平均而非按 token 平均：长度 3 与 1 的两条回答权重相同。
- 非法输入抛 `ValueError`：非张量 / 非浮点 / 含 NaN·Inf / logp 非二维 / 形状不符 / $N$ 或 $T<1$ / mask 非 bool 或形状不符 / 有空回答 / 优势形状或 dtype 不符 / $\epsilon\notin(0,1)$。
- CPU、确定性、不调用随机数。
