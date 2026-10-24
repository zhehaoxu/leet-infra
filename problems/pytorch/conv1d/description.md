# 一维卷积（conv1d）

## 题目描述

一维卷积把二维卷积沿一个空间（序列）维滑动：卷积核在长度方向上逐步移动，对每个输出位置
把所有输入通道与核对应元素相乘再求和。它广泛用于时序信号、1D-CNN、音频/文本特征提取。
这里的「卷积」沿用深度学习约定，即**互相关**（cross-correlation，核不翻转）。

## 规则

- 实现 `solve(x, weight, bias, stride=1, padding=0)`，`x` 形状 `(N, Cin, L)`。
- `weight` 形状 `(Cout, Cin, kL)`，`bias` 形状 `(Cout,)`（也可能为 `None`）。
- 输出长度 $L_{out} = \left\lfloor \dfrac{L + 2\cdot\text{padding} - k_L}{\text{stride}} \right\rfloor + 1$，
  输出形状 `(N, Cout, L_out)`。
- 每个输出元素：
  $$ \text{out}[n,o,t] = \text{bias}[o] + \sum_{ci=0}^{C_{in}-1} \sum_{j=0}^{k_L-1} x\big[n, ci, t\cdot\text{stride}+j\big]\cdot w[o, ci, j] $$
- 先在长度维按 `padding` 两侧补零，再按 `stride` 滑动。结果须与
  `F.conv1d(x, weight, bias, stride=stride, padding=padding)` 一致。

## 示例

```
输入：x = [[[1, 2, 3, 4, 5]]]           形状 (1,1,5)
     weight = [[[1, 2, 3]]]             形状 (1,1,3)
     bias = [0], stride=1, padding=0
   out[0] = 1*1 + 2*2 + 3*3 = 14
   out[1] = 2*1 + 3*2 + 4*3 = 20
   out[2] = 3*1 + 4*2 + 5*3 = 26
输出：[[[14, 20, 26]]]                  形状 (1,1,3)
```

## 约束

- float32 输入；`stride >= 1`，`padding >= 0`
- `weight` 的输入通道维必须等于 `x` 的通道数 `Cin`（本题不做分组）
- 不使用 `F.conv1d` / `torch.nn.Conv1d`；需手写滑动窗口
