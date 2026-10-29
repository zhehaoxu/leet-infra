# DPO 损失的解析反向

## 题目描述

**DPO**（Rafailov et al. 2023, *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, arXiv:2305.18290）的前向损失大家都熟：

$$L=-\frac{1}{B}\sum_b\log\sigma(x_b),\qquad x_b=\beta\Delta_b,\qquad \Delta_b=(s_w-\text{ref}_w)-(s_l-\text{ref}_l),$$

其中 $s_w,s_l$ 是 chosen/rejected 两条回答的**序列级**对数概率（逐 token logprob 在有效位置上求和）。平时我们把反向交给 `autograd`；但手写融合核、自定义 RL 循环、或要检查框架梯度是否正确时，**解析梯度**是不可替代的基准。

本题要求：给定逐 token 对数概率与掩码，返回损失，以及损失对两侧逐 token logprob 的解析梯度。答案是闭式的，一行链式法则就能推完，但符号、$1/B$ 的缩放与掩码位置必须精确。

## 规则

给定 $\text{logp\_w},\text{logp\_l}$ 形状 $(B,T)$、参考序列 logprob $\text{ref}_w,\text{ref}_l$ 形状 $(B,)$、两个掩码 $(B,T)$ bool（每条样本每侧至少一个 True）、$\beta>0$：

1. **序列对数概率**：$s_w=\sum_t\text{mask}_w\cdot\text{logp}_w$，$s_l$ 同理。
2. **隐式奖励差**：$\Delta=(s_w-\text{ref}_w)-(s_l-\text{ref}_l)$，$x=\beta\Delta$。
3. **前向**：$L=\frac{1}{B}\sum_b\mathrm{softplus}(-x_b)$（与 $-\log\sigma(x)$ 等价，必须用稳定式）。
4. **反向**：$\dfrac{\partial L}{\partial x_b}=-\sigma(-x_b)$，于是
   $$\frac{\partial L}{\partial\Delta_b}=-\frac{\beta\,\sigma(-x_b)}{B},\qquad
   \frac{\partial L}{\partial\text{logp\_w}[b,t]}=\frac{\partial L}{\partial\Delta_b}\,\text{mask}_w[b,t],\qquad
   \frac{\partial L}{\partial\text{logp\_l}[b,t]}=-\frac{\partial L}{\partial\Delta_b}\,\text{mask}_l[b,t].$$
   两侧符号相反：提高 rejected 的概率会把 $\Delta$ 压低、推高损失。

返回 `loss`（0 维 float64）、`grad_logp_w`、`grad_logp_l`（各 $(B,T)$ float64）。

## 示例

```
logp_w = [[-0.5, -0.4, -0.6], [-0.2, -0.3, -0.1]]     ref_w = [-1.5, -1.0]
logp_l = [[-0.7, -0.8, -0.9], [-0.4, -0.5, -0.6]]     ref_l = [-2.4, -1.5]
mask_w = 全 True
mask_l = [[True, True, False], [True, True, True]]
β = 0.1

第 0 条：s_w = -1.5, s_l = -1.5
        Δ = (-1.5 + 1.5) - (-1.5 + 2.4) = -0.9      x = -0.09
第 1 条：s_w = -0.6, s_l = -1.5
        Δ = (-0.6 + 1.0) - (-1.5 + 1.5) =  0.4      x =  0.04

loss = mean(softplus(0.09), softplus(-0.04)) = mean(0.739159, 0.673347) = 0.706253
dL/dΔ = -β·σ(-x)/B = [-0.026124, -0.024500]
grad_logp_w = [[-0.026124]×3, [-0.024500]×3]
grad_logp_l = [[ 0.026124,  0.026124, 0.0], [0.024500]×3]   # mask 位置严格为 0
```

## 约束

- 三个返回值均为 float64；`loss` 是 0 维标量，两个梯度是 $(B,T)$。
- 掩码位置梯度**严格为 0**，不能写成同一个常数；改动被 mask 位置的 logp 不得影响损失与其它位置的梯度。
- `1/B` 的均值缩放必须体现在梯度里：同一条样本放进更大的 batch，每个 token 的梯度会缩小。
- $\Delta=0$ 时 $x=0$，$L=\ln2$、$\partial L/\partial\Delta=-\beta/(2B)$（与 $\beta$ 成正比）。
- 非法输入抛 `ValueError`：非张量 / logp 非二维或形状不符 / 非浮点 / 含 NaN·Inf / 参考形状不是 $(B,)$ 或 dtype 不符 / 掩码非 bool 或形状不符 / 某条样本某一侧没有有效 token / $\beta\le0$。
- CPU、确定性、不调用随机数；实现可用 autograd 对拍验证，但返回值必须是解析式。
