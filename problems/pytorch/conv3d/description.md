# 三维卷积（conv3d）

## 题目描述

三维卷积在**三个**空间维 $(D, H, W)$ 上同时滑动卷积核，输入形如体素数据（voxel）、
视频帧序列或医学影像（CT/MRI）。每个输出元素是核与一个三维窗口在所有输入通道上的
逐元素乘积之和。同 conv1d/conv2d，这里的「卷积」是**互相关**（核不翻转）。

## 规则

- 实现 `solve(x, weight, bias, stride=1, padding=0)`，`x` 形状 `(N, Cin, D, H, W)`。
- `weight` 形状 `(Cout, Cin, kD, kH, kW)`，`bias` 形状 `(Cout,)`（也可能为 `None`）。
- 为简化，本题 `stride` 与 `padding` 为标量，三个空间维共用同一值。各维输出长度：
  $$ L_{out} = \left\lfloor \frac{L + 2\cdot\text{padding} - k_L}{\text{stride}} \right\rfloor + 1,\quad L \in \{D,H,W\} $$
- 每个输出元素：
  $$ \text{out}[n,o,a,b,c] = \text{bias}[o] + \sum_{ci}\sum_{di}\sum_{hi}\sum_{wi} x[n, ci, a\cdot s+di, b\cdot s+hi, c\cdot s+wi]\cdot w[o, ci, di, hi, wi] $$
- 先在三个空间维两侧各补 `padding` 个零，再按 `stride` 滑动。结果须与
  `F.conv3d(x, weight, bias, stride=stride, padding=padding)` 一致。

## 示例

```
输入：x = arange(1..8) reshape (1,1,2,2,2)   （体素值 1,2,...,8）
     weight = 全 1，形状 (1,1,2,2,2)
     bias = [0], stride=1, padding=0
   输出尺寸 (2-2)//1+1 = 1，整个 2x2x2 窗口求和 = 1+2+...+8 = 36
输出：[[[[[36]]]]]                            形状 (1,1,1,1,1)

若 padding=1：输出尺寸 (2+2-2)//1+1 = 3 → 形状 (1,1,3,3,3)
```

## 约束

- float32 输入；`stride >= 1`，`padding >= 0`（本题取标量）
- `weight` 的输入通道维必须等于 `x` 的通道数 `Cin`（不做分组）
- 不使用 `F.conv3d` / `torch.nn.Conv3d`；需手写三维滑动窗口
