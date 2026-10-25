# PEFT：IA3 逐元素激活重标定

## 题目描述

**IA³**（Infused by **I**nhibiting and **A**djusting with **A**t**t**ention，Liu et al., arXiv:2205.05638）是比 LoRA 更省参数的 PEFT：它不动权重矩阵，而是学几个**逐元素缩放向量**，把注意力/前馈里的**中间激活**按维乘性重标定（放大有用的维、抑制无关的维）。原始 IA³ 分别作用在 self-attention 的 key、value 与 FFN 中间层；本抽象到单层线性变换的输出重标定。

给定一个线性层输出后接一个 IA³ 向量 \(\lambda\)，前向为：

\[y = \lambda \odot (W x)\]

\(\lambda\) 逐输出维广播（对 batch 内每个样本、每个 token 用同一个 \(\lambda\)）。初始化 \(\lambda=\mathbf 1\)（恒等，不改动预训练）。

## 规则

给定输入 `x` 形状 \((\dots, d_\text{in})\)、权重 `W` 形状 \((d_\text{out}, d_\text{in})\)、IA³ 缩放向量 `lam` 形状 \((d_\text{out},)\)：

1. 线性变换 \(h = x\,W^\top\)（`nn.Linear` 惯例，形状 \((\dots, d_\text{out})\)）。
2. 逐输出维重标定 \(y = \lambda \odot h\)（\(\lambda\) 在最后维广播）。

返回 `y`（形状 \((\dots, d_\text{out})\)）。全程 float32，禁止调用现成 PEFT 库。

## 示例

\(W\)（3×5）同 LoRA 题；\(\lambda=[1.2,\ 0.8,\ 1.5]\)；`x`（2×5）同 LoRA 题：

```
y = [[0.516, 0.728, -1.020],
     [0.336, -0.256,  0.555]]
```

## 约束

- CPU float32；`x` 支持 2D 与 3D（最后维当特征）。
- \(\lambda\equiv\mathbf 1\) 时结果等于普通线性 \(xW^\top\)。
- 与「先把 \(\lambda\) 折进权重行 \((\lambda\odot W)\) 再矩阵乘」逐元素等价。
