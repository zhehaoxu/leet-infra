# One-Hot 编码

## 题目描述

把整数类别索引编码成独热向量：在末尾**新增一个类别维**，被索引的位置为 1，其余为 0。

$$ \text{onehot}(v)_c = \begin{cases} 1 & c = v \\ 0 & \text{否则} \end{cases} $$

## 规则

- 实现 `solve(indices, num_classes=None)`，输出形状 `(*indices.shape, num_classes)`，dtype 为 `int64`（`torch.long`）。
- `num_classes=None` 时取 $\max(\text{indices})+1$；显式给定时按给定值（多出的列补 0，若小于 $\max+1$ 则越界报错）。
- 与 `F.one_hot` 语义一致。

## 示例

```
输入：indices=[1,3,2]  → num_classes=max+1=4
输出：[[0,1,0,0],
      [0,0,0,1],
      [0,0,1,0]]
```

## 约束

- `indices` 为非负整数张量；每行恰好一个 1
