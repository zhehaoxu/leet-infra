# One-Hot 编码

给定一批类别标签，把它们转换成 one-hot 矩阵。这是分类损失、embedding 查表等场景的基础操作。

输入形状 `(N,)`、`dtype=int64` 的类别下标，返回形状 `(N, C)`、`dtype=float32` 的张量：第 `i` 行只有第 `labels[i]` 列为 `1.0`，其余为 `0.0`。

## 规则

1. 若存在 `label < 0` 或 `label >= num_classes`，抛出 `ValueError`。
2. `N = 0`（空批次）时返回形状 `(0, C)` 的空张量。
3. **禁止**使用 `torch.nn.functional.one_hot` / `F.one_hot`（评测与参考实现对照数值）。
4. 输出必须是独立新张量，不得与输入共享存储。

## 示例

```python
labels = torch.tensor([1, 0, 2])
solve(labels, 3)
# tensor([[0., 1., 0.],
#         [1., 0., 0.],
#         [0., 0., 1.]])
```

## 提示

- `torch.zeros` + 高级索引赋值是最直接的写法
- `scatter_` / `one_hot` 的底层就是 scatter：`out.scatter_(1, labels.unsqueeze(1), 1.0)`
- 范围校验可以用 `(labels >= 0) & (labels < num_classes)` 配合 `torch.all`

## 约束

- `0 <= N <= 4096`，`1 <= C <= 1000`
