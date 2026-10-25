# 对齐训练：PPO 截断价值损失

## 题目描述

PPO 除对策略做信任域截断外，价值函数更新也常用「截断 MSE」限制单步把 \(V_\theta\) 拉离旧值 \(V_\text{old}\) 太远（与策略 clip 对称，稳定 value head 训练）。给定回归目标 \(R\)（GAE 的 return），逐元素损失取「未截断」与「截断」两平方误差的**较大者**：

\[ \mathcal{L}^V = \frac1N\sum_i \max\!\Big((V_{\theta,i}-R_i)^2,\ \big(\operatorname{clip}(V_{\theta,i},\,V_{\text{old},i}-\epsilon,\,V_{\text{old},i}+\epsilon)-R_i\big)^2\Big) \]

clip 把新价值限制在 \([V_\text{old}-\epsilon,\,V_\text{old}+\epsilon]\)；取 max 使「越界且误差更大」的方向不被奖励削减（保守更新）。

## 规则

输入同形状 \((N,)\) 的 `values_new`（\(V_\theta\)）、`values_old`（\(V_\text{old}\)）、`returns`（\(R\)）、标量 `epsilon`。返回 0 维 float32 标量张量＝平均截断价值损失。禁止现成 RL 库。

## 示例

`epsilon=0.2`、\(V_\theta=[0.6,1.2,-0.3,0.9]\)、\(V_\text{old}=[0.5,1.0,-0.2,0.8]\)、\(R=[0.7,1.5,-0.5,0.8]\)：

```
clip(V_new, V_old±0.2) = [0.6, 1.2, -0.3, 0.9]   (本例未越界，clip 后与新值相同)
逐项 max((V_new-R)^2,(clip-R)^2) = [0.01, 0.09, 0.04, 0.01]
loss = mean = 0.0375
```

## 约束

- CPU float32；\(\epsilon>0\)。
- 两个平方误差取 `max`（不是 `min`、不是只算 clip 后）。
- 当 \(V_\theta\) 全在 \([V_\text{old}\pm\epsilon]\) 内时，clip 项与未 clip 项相等，损失退化为普通 MSE。
