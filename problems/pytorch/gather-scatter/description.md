# Gather 索引取值

## 题目描述

`gather` 沿指定维 `dim`，用 `index` 里的值逐元素地从 `input` 取数，输出形状与 `index` 相同。它是 top-k、按位置收集等场景的基础。

以三维、`dim=1` 为例：

$$ \text{out}[i][j][k] = \text{input}[i][\ \text{index}[i][j][k]\ ][k] $$

## 规则

- 实现 `solve(input, index, dim)`，`index` 与 `input` **维数相同**，`index` 在每个非 `dim` 维上长度 $\le$ `input` 对应维；`index` 为整数张量。
- 输出形状 = `index.shape`。
- **dim 语义陷阱**：只有 `dim` 这一根轴被 `index` 替换取值，其余轴保持自身坐标。例如 `dim=0` 时 $\text{out}[i][j][k]=\text{input}[\text{index}[i][j][k]][j][k]$。
- 支持负 `dim`（`-1` 即最后一维）。
- 与 `torch.gather` 完全一致。

## 示例

```
输入：input=[[10,20,30],[40,50,60]], index=[[2,0,1]], dim=1
   第0行按 index [2,0,1] 取列 → [30,10,20]
输出：[[30,10,20]]
```

## 约束

- float32/float64 数据 + int64 索引
- `index` 各值在对应维范围内
