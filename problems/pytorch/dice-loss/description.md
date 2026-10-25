# Soft Dice 损失

## 题目描述

分割任务里常用 **Dice 系数**衡量预测掩膜与真值掩膜的重叠：\(\text{Dice}=\frac{2|X\cap Y|}{|X|+|Y|}\)。它对小目标/正类样本不敏感的问题比逐像素交叉熵更友好。**Soft Dice 损失**把硬 0/1 掩膜换成连续概率 \(p\in[0,1]\)，使其可微：

\[ \mathcal{L}_{\text{dice}} = 1 - \frac{2\sum_c p_c y_c + s}{\sum_c p_c + \sum_c y_c + s} \]

分子分母的交并关系直接来自 V-Net / MONAI `DiceLoss`。加平滑项 \(s\) 防全零除。

## 规则

实现 `solve(prob, target, smoothing=1.0, reduction="mean")`。`prob`（float32，\((N, C)\)，取值 \([0,1]\) 的预测概率）、`target`（float32，\((N, C)\)，取值 \(\{0,1\}\) 的软/硬标签）。**逐样本**沿最后一维 \(C\) 求：\(\text{inter}=\sum_c p_c y_c\)、\(\text{denom}=\sum_c p_c+\sum_c y_c\)，\(loss_i = 1-\frac{2\,\text{inter}+s}{\text{denom}+s}\)。按 `reduction`（`"mean"`/`"sum"`/`"none"`）归约。非法 reduction 抛 `ValueError`。

## 示例

\(N=1, C=2\)，`prob=[[0.9,0.1]]`，`target=[[1,0]]`，`smoothing=1.0`：

```
inter = 0.9*1 + 0.1*0 = 0.9
denom = (0.9+0.1) + (1+0) = 2.0
dice  = (2*0.9 + 1) / (2.0 + 1) = 2.8/3 = 0.933333
loss  = 1 - 0.933333 = 0.0666667
```

## 约束

- CPU float32；`prob` 已在 \([0,1]\)（不要求沿 C 归一），`target` 为 0/1。
- 求和沿**最后一维**逐样本进行，再对 batch 归约。
- `smoothing`（\(s\)）加在分子与分母上，防除零。
- 完美预测 \(p=y\)（且非全零）时 dice=1、loss=0；全零样本由 \(s\) 兜底给 loss=0。
