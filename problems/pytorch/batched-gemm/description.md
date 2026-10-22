# 批量矩阵乘（BMM / matmul 语义）

## 题目描述

批量 GEMM 是 Attention（$QK^\top$、$\text{attn}V$）的计算骨架。给定 $A: (\dots, n, m)$ 与 $B: (\dots, m, p)$，按 `torch.matmul` 的广播语义：

$$
\text{out}[\dots, i, j] = \sum_{k=0}^{m-1} A[\dots, i, k]\cdot B[\dots, k, j]
$$

前面的批维度 $(\dots)$ 按广播规则对齐：可以完全相同、一侧为空（2D 矩阵）、或含长度 1 的维。

## 规则

- 实现 `solve(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor`，两者均 $\ge 2$ 维，内维必须相等（否则抛 `ValueError`）。
- **禁止调用 `torch.matmul` / `bmm` / `mm` / `@` 运算符 / `einsum` / `dot`**——回到定义：把 $a$ 扩出 $(\dots, n, m, 1)$、$b$ 扩出 $(\dots, 1, m, p)$，逐元素相乘后沿 $m$ 求和。
- 批维广播用 `torch.broadcast_shapes` + `expand` 处理；`expand` 出来的视图 stride 为 0，逐元素乘完全兼容。
- **陷阱**：输入可能是非连续的转置视图（如 `x.transpose(1,2)`），依赖 reshape/contiguous 的写法会在这里翻车；广播语义要覆盖「2D @ 3D」和「batch 维含 1」。

## 示例

```
输入：a = [[1,2],[3,4]], b = [[5,6],[7,8]]
输出：[[19, 22], [43, 50]]     # 1*5+2*7=19, 1*6+2*8=22, ...
```

## 约束

- float32/float64；测试张量较小（外积法内存 $O(nmp)$ 可接受）
- 与 `torch.matmul(a, b)` 逐位一致（atol 1e-5）
