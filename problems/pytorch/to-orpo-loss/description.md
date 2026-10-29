# ORPO：无参考模型的赔率比偏好优化

## 题目描述

**ORPO**（Hong et al. 2024, *ORPO: Monolithic Preference Optimization without Reference Model*, arXiv:2403.07691）把监督微调与偏好对齐压进**一个**训练阶段：不需要参考模型、不需要先做 SFT 再对齐，直接在偏好数据上同时做「提高 chosen 的似然」与「拉开 chosen/rejected 的赔率」。

DPO 用参考模型的 logprob 当锚点来约束漂移；ORPO 换了个思路——用 **chosen 自身的 SFT 项**当正则，再用**赔率比**（odds ratio）代替概率比做偏好项。赔率 $p/(1-p)$ 对 $p$ 接近 1 的样本更敏感：当模型已经把 chosen 的概率推到很高时，赔率的差异会被放大，从而持续施加梯度；这正好补上了 sigmoid 型偏好损失在「已分对」后梯度饱和的短板。

本题给定四条**已按长度归好**的序列级量（两条序列 logprob + 两个长度），只需组装 SFT 项与赔率项。

## 规则

给定长度 $N\ge 1$ 的批次：

1. **先做长度归一**：$\ell_w=\text{logp}_w/|\text{y}_w|$，$\ell_l=\text{logp}_l/|\text{y}_l|$。两者必须严格小于 0，否则 $p=e^{\ell}$ 不是合法概率、$\log(1-p)$ 无定义。
2. **赔率对数**必须用稳定式：
   $$\mathrm{logodds}(\ell)=\ell-\log\!\big(1-e^{\ell}\big)=\ell-\mathrm{log1p}(-e^{\ell}),$$
   不允许先算 $p$ 再取 $\log\frac{p}{1-p}$（$p\to 1$ 时会精度崩塌）。
3. **SFT 项**（只惩罚偏好回答的 NLL）：$L_\text{SFT}=-\frac{1}{N}\sum \ell_w$。
4. **赔率项**：$L_\text{OR}=-\frac{1}{N}\sum \log\sigma\big(\mathrm{logodds}(\ell_w)-\mathrm{logodds}(\ell_l)\big)$，log-sigmoid 用稳定式。
5. **总损失**：$L=L_\text{SFT}+\lambda\,L_\text{OR}$，$\lambda\ge 0$ 默认 $0.1$；不乘 $\beta$，也没有参考模型项。

返回 0 维 float64 标量。全程用 float64 累加。

## 示例

```
logp_w = [-2.0, -1.5]   |y_w| = [2, 3]
logp_l = [-3.0, -2.4]   |y_l| = [3, 4]
λ = 0.1

长度归一： ℓ_w = [-1.0, -0.5]      ℓ_l = [-1.0, -0.6]
赔率对数： logodds(ℓ_w) = [-0.541325, 0.432752]
           logodds(ℓ_l) = [-0.541325, 0.195870]
赔率差：   d = [0.0, 0.236882]
SFT 项：   -mean(ℓ_w) = 0.75
赔率项：   -mean(logsigmoid(d)) = 0.637426
总损失：   0.75 + 0.1 × 0.637426 = 0.813743
```

注意第 2 条样本若把赔率差错写成 $\ell_w-\ell_l=0.1$，赔率项会变成另一组数——两者差远超 $10^{-3}$，足以让实现被判错。

## 约束

- 返回 0 维 float64；输入可以是 float32，内部必须升到 float64。
- $\lambda=0$ 时严格退化为 SFT；两侧**平均**对数概率相同时赔率差为 0，赔率项恰为 $\ln 2$。
- 对长度归一敏感：同一条序列 logprob 摊到更长的回答上，$\ell$ 会变、两项都随之改变。
- 非法输入抛 `ValueError`：非张量 / 非一维 / 非浮点 logp / 非整型长度 / 形状不一致 / 长度 < 1 / 含 NaN·Inf / $\lambda<0$ / 长度归一后 $\ge 0$。
- CPU、确定性、不调用随机数。
