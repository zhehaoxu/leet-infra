# Label Smoothing：带标签平滑的交叉熵

## 题目描述

one-hot 目标把概率质量全压到正确类，会驱使模型输出**极端自信**的 logit（softmax 饱和、梯度消失、校准变差、易过拟合）。**Label Smoothing**（Szegedy et al. 2016, R-SLU）把硬标签 $y$ 换成"软标签"：给正确类留 $1-\alpha$，把 $\alpha$ 平均分摊到全部 $V$ 类上。交叉熵 $\mathcal{L}=-\sum_v q_v\log p_v$（$q$ 为目标分布、$p$ 为预测 softmax）随之拆成两项：

$$\mathcal{L}_\text{LS} = (1-\alpha)\underbrace{\big(-\log p_{y}\big)}_{\text{标准 NLL}} + \alpha\underbrace{\Big(-\tfrac{1}{V}\sum_v \log p_v\Big)}_{\text{对均匀分布的正则项}}$$

第二项等价于 $\alpha\cdot\mathrm{KL}(\text{uniform}\,\|\,p)$ 加上一个常数——它把预测往均匀分布轻推，抑制过度自信。这正是它与 `kl_div` 的呼应点。

## 规则

给定 `logits` $(N, V)$、`target` $(N,)$（每行正确类下标）、标量 `alpha` $\alpha$：

1. $\log p = $ `log_softmax(logits, dim=-1)`（数值稳定，先减最大值）
2. NLL 项：$u_n = -\log p[n, $`target[n]`$]$
3. 平滑项：$s_n = -\frac{1}{V}\sum_v \log p[n, v]$（对全类 $-\log p$ 取平均）
4. 逐样本：$(1-\alpha)\,u_n + \alpha\, s_n$
5. 对 batch 取均值

返回 0 维标量 `torch.Tensor`＝ 平均 label-smoothing loss。要求 $\alpha=0$ 时精确退化为标准交叉熵。

## 示例

```
alpha = 0.1, V = 3
logits = [[2.0,1.0,0.1],[0.5,2.5,1.0],[1.0,1.0,3.0]]
target = [0, 1, 2]

log_softmax 后逐行：
  行0  logp≈[-0.7867,-1.7867,-2.6867]  NLL=0.7867  平滑均值=(0.7867+1.7867+2.6867)/3=1.7534
  行1  logp≈[-2.3133,-0.3133,-1.8133]  NLL=0.3133  平滑均值=1.4800
  行2  logp≈[-2.4076,-2.4076,-0.4076]  NLL=0.4076  平滑均值=1.7409
逐样本 0.9*NLL + 0.1*平滑 = [0.8834, 0.4299, 0.5409]
mean = 0.436532
（alpha=0 时 mean = 0.320977，等于 nn.functional.cross_entropy）
```

## 约束

- CPU float32；$\alpha \in [0,1)$；`target` 为合法类下标。
- 必须用稳定 log-softmax（不减最大值的 `log(softmax)` 会溢出）。
- $\alpha=0$ 与 `F.cross_entropy(..., reduction='mean')` 逐位一致；返回 0 维标量张量。
