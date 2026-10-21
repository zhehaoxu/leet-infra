# autograd 求导入门

## 题目描述

PyTorch 的自动微分（autograd）记录前向计算构成的动态图，`backward()` 时沿图反向传播梯度。掌握「前向 → `backward` → 读 `.grad`」的流程是写任何训练循环的前提。

## 规则

实现 `solve(x: torch.Tensor, expr: str) -> torch.Tensor`，**用 autograd**（前向构图 + `.backward()`）返回标量目标对 `x` 的梯度 $\dfrac{\mathrm{d}\sum y}{\mathrm{d}x}$（逐元素梯度，形状与 `x` 相同）：

| expr | 前向目标 $y$（逐元素） | 解析梯度 |
|---|---|---|
| `"quadratic"` | $3x^2 + 2x$ | $6x + 2$ |
| `"cubic"` | $x^3$ | $3x^2$ |
| `"exp"` | $e^x$ | $e^x$ |

关键要求：

1. **不许手代解析公式**——必须走 `backward()`（本题不强制检查实现方式，但请诚实练习）。
2. **不得污染输入**：传入的 `x` 可能未开启梯度追踪（`requires_grad=False`）。直接对入参调 `requires_grad_(True)` 会弄脏调用方的张量；正确做法是先 `x.detach().requires_grad_(True)` 再造叶子。
3. `expr` 非法时抛 `ValueError`。

## 示例

```
输入：x = [1.0, 2.0], expr = "quadratic"
输出：[8.0, 14.0]        # 6*1+2, 6*2+2
```

## 约束

- `x` 为 float32 张量，任意形状
- 返回的梯度张量本身不应携带计算图（`leaf.grad` 天然满足）
