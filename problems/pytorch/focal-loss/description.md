# Focal 损失

## 题目描述

标准交叉熵对每个样本一视同仁，但在类别极不平衡（如目标检测里背景远多于前景）时，大量「 easy 负样本」主导梯度。RetinaNet（Lin et al., arXiv:1708.02002）提出 **Focal Loss**：给交叉熵乘一个调制因子 \((1-p_t)^\gamma\)，把「已经预测对（\(p_t\) 大）」的样本损失压低、聚焦到难样本。

\[ \text{FL}(p_t) = -\,\alpha_t\,(1-p_t)^\gamma \log(p_t),\quad p_t = \text{softmax}(\text{logits})[\text{target}] \]

## 规则

实现 `solve(logits, target, gamma=2.0, alpha=None, reduction="mean")`。`logits`（float32，\((N, C)\)）为未归一化类别打分，`target`（int64，\((N,)\)）为真实类别下标。令 \(p_t\) 为 softmax 后真实类的概率、\(\log p_t\) 为其对数，逐样本损失为 \(-(1-p_t)^\gamma \log p_t\)；若 `alpha` 不为 `None`（一个标量）则再乘 `alpha`。按 `reduction`（`"mean"`/`"sum"`/`"none"`）归约。`reduction` 非法须抛 `ValueError`。

## 示例

\(N=2, C=2\)，`logits=[[2.0,0.5],[0.1,1.0]]`，`target=[0,1]`，`gamma=2.0, alpha=None`：

```
行0: p_t=softmax[0]=0.81757 -> -(1-0.81757)^2*log(0.81757)=0.0067029
行1: p_t=softmax[1]=0.71092 -> -(1-0.71092)^2*log(0.71092)=0.0285035
mean = 0.0176031645
```

## 约束

- CPU float32；\(p_t\) 取真实类 softmax 概率，`log(p_t)` 用数值稳定的 `log_softmax`。
- `gamma=0` 且 `alpha=None` 时，Focal 完全退化为标准交叉熵（应与 `cross_entropy` 逐样本 NLL 一致）。
- `alpha` 为标量权重或 `None`。
- 支持 `reduction` 三种取值，其它字符串抛 `ValueError`。
