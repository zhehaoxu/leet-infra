# SELU 自归一化激活（SELU）

## 题目描述

SELU（Scaled Exponential Linear Unit）是带固定缩放的 ELU 变体。它的特别之处不在曲线形状，
而在两个**魔法常数** $\lambda$（scale）与 $\alpha$：只要权重按配套方案初始化，一连串 SELU 层
会把每层激活自动拉回均值 0、方差 1——即「自归一化」，无需额外的 BatchNorm/LayerNorm。

## 规则

- 实现 `solve(x)`，逐元素返回：
  $$ \operatorname{SELU}(x)=\begin{cases}\lambda\,x & x>0\\ \lambda\,\alpha\,(e^{x}-1) & x\le 0\end{cases} $$
- 常数：$\alpha=1.6732632423543772848170425930144$，$\lambda=1.0507009873554804977367086399070$。
- 形状不变、保持 dtype；结果须与 `F.selu(x)` 一致。

## 示例

```
输入：x = [-2.0, -0.5, 0.0, 0.5, 2.0, 7.0]
输出：[-1.52017, -0.69176, 0.0, 0.52535, 2.1014, 7.35491]
（x>0 时约放大 1.05 倍；x<=0 时为负的饱和指数，0 处连续为 0）
```

## 约束

- 支持 float32 / float64 输入，输出 dtype 与输入一致
- $\lambda,\alpha$ 必须用题面给出的高精度值，否则与 `F.selu` 偏离
