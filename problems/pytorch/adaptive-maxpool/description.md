# 自适应最大池化（Adaptive Max Pooling）

## 题目描述

普通 `maxpool2d` 靠固定 `kernel/stride` 决定输出尺寸，输入一变输出就跟着变；
**自适应**池化反过来——你指定**想要的输出尺寸** `output_size`，框架自动反推每个输出位
该覆盖输入的哪一段，并在段内取最大值。它常用来把任意尺寸的特征图压成固定大小
（如分类头前的 `AdaptiveMaxPool2d(1)`），或与 `adaptive-avgpool` 互补地保留显著激活。

## 规则

- 实现 `solve(x, output_size)`，`x` 形状 `(N, C, H, W)`，`output_size` 为整数或 `(oh, ow)`。
- 每个输出位覆盖的输入段边界沿用 PyTorch 的 floor/ceil 公式：
  第 `i` 个输出对应输入区间 `[floor(i*Hin/outH), ceil((i+1)*Hin/outH))`（宽同理）。
- 段内所有元素取 `max`，输出形状 `(N, C, oh, ow)`。
- 边界切法与 `adaptive-avgpool` **完全相同**，只是聚合算子由均值换成最大值。

## 示例

```
输入：x = [[ 0,  1,  2,  3],
          [ 4,  5,  6,  7],
          [ 8,  9, 10, 11],
          [12, 13, 14, 15]]，output_size = 2
   四个 2x2 段分别取 max：max{0,1,4,5}=5, max{2,3,6,7}=7,
                         max{8,9,12,13}=13, max{10,11,14,15}=15
输出：[[ 5,  7],
      [13, 15]]
```

## 约束

- float32 输入；`output_size` 各维须为正且不大于对应输入尺寸（一般成立）
- 当 `output_size == 1` 时结果即整图（逐通道）最大值，等价 global max pooling
