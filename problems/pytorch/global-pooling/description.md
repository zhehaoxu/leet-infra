# 全局池化（Global Avg/Max Pool）

## 题目描述

全局池化把每个通道的整张特征图压成一个数：GAP 取均值、GMP 取最大值，输出恒为 $(N, C, 1, 1)$。它是 ResNet 末层「任意分辨率进、定长向量出」的关键，也常用于 CAM 可解释性。

$$
\text{GAP}[n,c] = \frac{1}{HW}\sum_{i,j} x[n,c,i,j],\qquad \text{GMP}[n,c] = \max_{i,j} x[n,c,i,j]
$$

## 规则

- 实现 `solve(x: torch.Tensor, mode: str = "avg") -> torch.Tensor`，x 形状 `(N, C, *)`（`*` 为任意个空间/序列维）。
- 归约维是 `dim >= 2` 的所有维，且 **`keepdim=True`**——保持 `(N, C, 1, ...)` 让后续 `flatten`/广播语义清晰。
- `mode` 取 `"avg"` 或 `"max"`，其他值抛 `ValueError`。
- 归约不跨通道：每个 $(n,c)$ 独立归约。

## 示例

```
输入：x 形状 (1,2,2,2) = [[[1,2],[3,4]], [[5,6],[7,8]]]
mode="avg" → (1,2,1,1) = [[2.5], [6.5]]
mode="max" → (1,2,1,1) = [[4.0], [8.0]]
```

## 约束

- float32/float64
- avg 与 `F.adaptive_avg_pool2d(x, 1)` 一致；max 与 `F.adaptive_max_pool2d(x, 1)` 一致
