# softmax 反向：给定输出与上游梯度求 logits 梯度

## 题目描述

softmax 是被反向穿过最多次的归一化算子：注意力打分之后、任何「先归一化再加权」的地方之后。它的雅可比是 \(n\times n\) 的 \(\partial y_i/\partial x_j = y_i(\delta_{ij}-y_j)\)，但真正的实现从不构造这个矩阵——沿归一化轴做一次点积就够了：

$$dx_i = y_i\Big(grad_i-\sum_k y_k\,grad_k\Big)$$

本题只给你前向输出 \(y\)（softmax 或 log_softmax）与上游梯度 \(grad\)，要求还原 \(dx\)。这一步也是 FlashAttention 反向里唯一需要「重算」的量。

## 规则

- 完整签名与参数含义只写在 starter 的 docstring 里，那是唯一判题口径。
- 禁止显式构造 \(n\times n\) 雅可比矩阵再乘 grad；只允许沿 `dim` 的一次归约。
- `log_input=True` 表示前向是 log_softmax，公式变为 \(dx_i = grad_i - e^{y_i}\sum_k grad_k\)，与 softmax 支路**不可互换**。
- 若 grad 沿 `dim` 是常数（例如全 1），softmax 支路必须给出全 0——这就是 softmax 对 logits 整体平移的不变性。
- 返回与 \(y\) 同形、同 dtype；\(y\)/grad 形状不符、`dim` 越界、`log_input` 非 bool 一律抛 ValueError。

## 示例

沿末轴归一化，数值取自 test_cases.py 回填的 EXAMPLE_* 常量：

```
y    = [[0.5, 0.25, 0.25],   grad = [[1.0, 2.0, 3.0],
        [0.25, 0.5, 0.25]]           [0.0, 1.0, 0.0]]
sum(y*grad) per row = [1.75, 0.5]
dx   = [[-0.375, 0.0625, 0.3125],
        [-0.125, 0.25,   -0.125]]
```

第一行逐项：\(0.5(1-1.75)=-0.375\)、\(0.25(2-1.75)=0.0625\)、\(0.25(3-1.75)=0.3125\)。

## 约束

- 输入可为 float32 或 float64；与 `torch.autograd.grad` 对拍，float64 用例 atol=1e-9、float32 用例 atol=1e-6
- 支持任意 `dim`（含中间轴与负数下标）以及 1 维输入；内部用 float64 累计后折回原 dtype
