# 转置卷积 / 反卷积（ConvTranspose2d）

## 题目描述

转置卷积常被称为「反卷积」，用于把特征图**上采样**变大（语义分割解码头、生成器、可微缩放）。
它不是普通卷积的逆，而是普通卷积**对输入求梯度**的那一步：每个输入元素把自己的值乘以核，
**散射累加**（scatter-add）到若干输出位置。

$$ o_h = (H-1)\cdot \text{stride} - 2\cdot\text{padding} + k_h $$

## 规则

- 实现 `solve(x, weight, bias, stride, padding)`，`x` 形状 `(N, Cin, H, W)`。
- **注意 weight 约定与 conv2d 相反**：形状 `(Cin, Cout, kh, kw)`。
- 对每个输入元素 `x[n,ci,iy,ix]`，把 `weight[ci,:,ky,kx]` 累加到输出位置
  `[n,:, iy*stride - padding + ky, ix*stride - padding + kx]`，**越界则丢弃**。
- 最后按 `Cout` 加 bias。输出形状 `(N, Cout, o_h, o_w)`（见上式）。

## 示例

```
输入：x=[[[[1,2],[3,4]]]] (1,1,2,2), weight=全1 (1,1,2,2), bias=None, stride=1, padding=0
      每个输入值把它 2x2 邻域的输出各加一次，重叠处累加
输出：[[[[1, 3, 2],
        [4, 10, 6],
        [3, 7, 4]]]]   (1,1,3,3)
      （中心 10 = 1+2+3+4，正是四个输入都覆盖到该输出位置）
```

## 约束

- float32 输入；`stride >= 1`，`padding >= 0`；`bias` 可为 `None` 或形状 `(Cout,)`
- 需支持 `stride ∈ {1,2}`、`padding ∈ {0,1}` 的各种组合且与 `F.conv_transpose2d` 一致
