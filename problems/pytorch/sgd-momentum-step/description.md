# SGD 动量更新单步（SGD + Momentum Step）

## 题目描述

SGD 是深度学习的基线优化器，加上**动量（momentum）**后能沿下降方向积累惯性、抑制震荡；再叠上**权重衰减（weight_decay）**得到 L2 正则版 SGD。本手写单次更新——给定旧参数、当前梯度、旧缓冲区，返回新参数与新缓冲区。

## 规则

- 实现 `solve(params, grads, buffers, lr, momentum=0.0, weight_decay=0.0)`，返回 `(new_params, new_buffers)`。
- **严格按 `torch.optim.SGD`（dampening=0、nesterov=False）的更新式**：

  1. 权重衰减并入梯度：$g \leftarrow g + \text{wd}\cdot p$
  2. 动量缓冲：$b \leftarrow \mu\, b + g$（momentum=0 时不启用缓冲）
  3. 参数更新：$p \leftarrow p - \text{lr}\cdot b$（momentum=0 时直接用 $g$ 代替 $b$）

- 入参不可变（要 clone），返回新张量。
- `buffers` 初始化为全零（PyTorch 首步懒建 buffer 也是零）。

## 示例

```
输入：params=[1.0, 1.0, 1.0]，grads=[0.5, -0.5, 2.0]，buffers=[0, 0, 0]
      lr=0.1，momentum=0.9，weight_decay=0.0
new_params=[0.95, 1.05, 0.80]
new_buffers=[0.5, -0.5, 2.0]
（首步：buf = 0.9*0 + g = g；p' = p - 0.1*buf）
```

## 约束

- 支持 float32 / float64；输出 dtype 与 `params` 一致
- 不使用 `torch.optim.SGD`；需手写更新式；须与 PyTorch 逐步完全一致（对拍见 notes）
