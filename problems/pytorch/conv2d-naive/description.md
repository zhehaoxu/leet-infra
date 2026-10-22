# 二维卷积前向（朴素实现）

## 题目描述

卷积是 CNN 的核心：每个输出位置、每个输出通道，都是输入感受野与卷积核的逐元素乘加（严格说是**互相关**，PyTorch 的 `conv2d` 就是不做翻转的）：

$$
\text{out}[n,o,i,j] = \sum_{c=0}^{C_{in}-1}\ \sum_{u=0}^{k_h-1}\ \sum_{v=0}^{k_w-1} x_{\text{pad}}[n,c,\,i\cdot s_h+u,\,j\cdot s_w+v]\cdot w[o,c,u,v] + \text{bias}[o]
$$

输出尺寸 $\left\lfloor \frac{H + 2p - k_h}{s_h} \right\rfloor + 1$。

## 规则

- 实现 `solve(x, weight, bias=None, stride=1, padding=0)`：x 形状 $(N, C_{in}, H, W)$，weight 形状 $(C_{out}, C_{in}, k_h, k_w)$。
- 先按 padding 补 **0**，再按公式累加。本题允许三重循环（输出位 × 输出通道，通道内用张量切片求和即可），不要求性能。
- 禁止调用 `F.conv2d` / `nn.Conv2d` / `conv_transpose` 等现成卷积。
- **陷阱**：weight 第 0 维是输出通道、第 1 维是输入通道——把它转置用反是手写第一坑；padding 是补 0（和 maxpool 的 $-\infty$ 相反）。

## 示例

```
输入：x = 单通道 3x3 = [[1,2,3],[4,5,6],[7,8,9]], weight = 1个 2x2 全 1 核
输出：[[[12, 16], [24, 28]]]     # 每个 2x2 窗口求和：1+2+4+5=12, ...
```

## 约束

- float32/float64；测试输入较小（朴素循环可承受）
- 结果与 `torch.nn.functional.conv2d` 一致
