# 数值稳定 Softmax

实现一个数值稳定的 softmax，支持可选的 mask 与任意归约维度。

$$
\text{softmax}(x)_i = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}
$$

先减去最大值可防止 `exp` 在浮点下溢出（`float32` 时 `exp(89)` 即为 `inf`）。

## 规则

1. 每一组（沿 `dim`）在被 mask 后的位置上重新归一化，即有效位置概率和为 `1`，无效位置为 `0`。
2. 若某一组全部被 mask（全 `False`），该组输出**全 0**（不允许出现 `nan`）。
3. 输出形状与 `logits` 相同，dtype 与输入一致。
4. 不允许直接调用 `torch.softmax` / `F.softmax`（评测会与参考实现对照数值）。

## 示例

```python
logits = torch.tensor([[2.0, 1.0, 0.5]])
solve(logits)
# tensor([[0.6286, 0.2312, 0.1403]])

mask = torch.tensor([[True, False, True]])
solve(logits, mask)
# tensor([[0.8176, 0.0000, 0.1824]])
```

## 提示

- `masked_fill` 配合 `keepdim=True` 的 `max` 可以广播处理
- 被 mask 的位置在求 `max` 时应填充 `-inf`，在求 `sum(exp)` 时应填充 `0`
- 注意 `-inf - (-inf)` 会产生 `nan`，全 mask 组需要特殊处理

## 约束

- `logits` 元素范围 `[-1e4, 1e4]`（必须处理溢出）
- `logits` 至少 2 维或按任意 `dim` 均可
