# GLU（门控线性单元）

## 题目描述

Gated Linear Unit 把输入沿某一维对半分成两块，一块作「值」、一块经 sigmoid 作「门」：

$$
\text{GLU}(x) = a \odot \sigma(b), \quad a, b = \text{split}_{dim}(x, 2)
$$

即把 `x` 沿 `dim` 均分为前后两半 $a$、$b$，输出 $a \cdot \text{sigmoid}(b)$，输出在该维减半。Transformer 的 FFN（如 SwiGLU）广泛使用门控结构。

## 规则

- 实现 `solve(x: torch.Tensor, dim: int = -1) -> torch.Tensor`。
- 沿 `dim` 用 `torch.chunk(x, 2, dim)` 对半分；前半 $\times$ `sigmoid(后半)`。
- 输出在 `dim` 上的长度为输入的一半；向量化实现。
- `dim` 维长度为奇数时无法均分，应抛异常（`torch.chunk` 本身会报错，无需特殊处理）。

## 示例

```
输入：x = [1.0, 2.0, 3.0, 4.0], dim = -1
拆分：a = [1.0, 2.0], b = [3.0, 4.0]
输出：[0.95257413, 1.96402752]   # 1*σ(3), 2*σ(4)
```

## 约束

- 输入为 float32/float64 张量，`dim` 指定维长度必须为偶数
- 结果与 `torch.nn.functional.glu` 一致
