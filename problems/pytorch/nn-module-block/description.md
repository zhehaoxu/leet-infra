# 组装 MLP Block

## 题目描述

`nn.Module` 是 PyTorch 组织模型的基本单元：在 `__init__` 里声明子模块与参数，在 `forward` 里描述数据流。框架自动收集参数、追踪梯度，`model.parameters()` 即可交给优化器。

## 规则

实现 `solve(in_features: int, hidden_multiplier: int, out_features: int) -> torch.nn.Module`，返回一个 **两层 MLP block** 模块实例：

$$\text{Linear}(in \to in \cdot m) \;\rightarrow\; \text{ReLU} \;\rightarrow\; \text{Linear}(in \cdot m \to out)$$

要求：

1. 用 `nn.Module` 子类实现（或 `nn.Sequential` 组装），内部只能使用 `nn.Linear` 与 `nn.ReLU`。
2. 隐藏层宽度为 `in_features * hidden_multiplier`。
3. `hidden_multiplier < 1` 时抛 `ValueError`。
4. 返回的模块必须已完成参数初始化（可直接前向）。

## 示例

```
输入：in_features=8, hidden_multiplier=4, out_features=2
模块：Linear(8→32) + ReLU + Linear(32→2)
可训练参数量：8*32+32 + 32*2+2 = 354
```

参数量核对：第一层权重 $8 \times 32$ + 偏置 $32$，第二层权重 $32 \times 2$ + 偏置 $2$。

## 约束

- `in_features, out_features ∈ [1, 256]`，`hidden_multiplier ∈ [1, 8]`
- 前向输入为 float32、形状 `(B, in_features)` 的批
- 评测在固定 seed（`torch.manual_seed(42)`）下实例化并检查，你的实现必须保证同 seed 同结果——**不要在自己的代码里重新 set seed**
