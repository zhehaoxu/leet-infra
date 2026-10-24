# DPO：Direct Preference Optimization 损失

## 题目描述

**DPO**（Rafailov et al. 2023）跳过"先训奖励模型再 RL"这一整套 RLHF 流程，把偏好数据直接当成一个分类目标来优化策略模型。核心洞察：RLHF 的最优策略满足 $\pi^*(y|x)\propto\pi_\text{ref}(y|x)\exp(r(y,x)/\beta)$，反过来可用策略与参考策略的对数似然比**隐式**表达奖励，于是奖励模型被消掉，剩下一条对偏好对 $(y_w \succ y_l)$ 的 Bradley–Terry 式 logistic 损失：

$$\mathcal{L}_\text{DPO} = -\log\sigma\!\Big(\beta\big[(\log\tfrac{\pi_\theta(y_w)}{\pi_\text{ref}(y_w)}) - (\log\tfrac{\pi_\theta(y_l)}{\pi_\text{ref}(y_l)})\big]\Big)$$

直觉：让" chosen 相对参考模型的改进量" 比 "rejected 相对参考模型的改进量" 更大。$\beta$ 控制偏离参考模型的强度。本题直接给定四个**序列级对数概率**（即整条回答各 token logprob 之和），只需组装出上式。

## 规则

给定一 batch（长度 $N$）的：`chosen_logp`、`chosen_ref_logp`、`rejected_logp`、`rejected_ref_logp`（均 $(N,)$ float），标量 `beta` $\beta$：

1. chosen 隐式奖励项 $= $ `chosen_logp` $-$ `chosen_ref_logp`
2. rejected 隐式奖励项 $= $ `rejected_logp` $-$ `rejected_ref_logp`
3. logits $= \beta\times($ ① $-$ ② $)$，shape $(N,)$
4. 逐样本 loss $= -\log\sigma(\text{logit})$，用**稳定** log-sigmoid：$\log\sigma(x)=\min(0,x)-\log(1+e^{-|x|})$
5. 对 batch 取**均值**

返回标量 `torch.Tensor`（0 维）＝ batch 平均 DPO loss。

## 示例

```
beta = 0.1
chosen_logp       = [-1.2, -0.8, -2.0]
chosen_ref_logp   = [-1.0, -1.0, -1.5]
rejected_logp     = [-2.5, -1.9, -1.0]
rejected_ref_logp = [-1.4, -1.6, -1.2]

① = [ -0.2, 0.2, -0.5 ]
② = [ -1.1, -0.3,  0.2 ]
logit = 0.1 * (① - ②) = 0.1*[0.9, 0.5, -0.7] = [0.09, 0.05, -0.07]
loss  = -mean(logsigmoid([0.09,0.05,-0.07])) = 0.682126
```

## 约束

- CPU float32；输入为序列级 logprob（已是各 token 之和，无需再累加）。
- log-sigmoid 必须数值稳定（等价 `F.logsigmoid`），不得直接 `log(sigmoid(x))` 造成下溢。
- 返回 0 维标量张量。
