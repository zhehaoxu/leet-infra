# Embedding 查表（含 padding_idx 置零）

## 题目描述

Embedding 层把离散的整数索引映射为稠密向量：本质是按索引从权重矩阵 $W \in \mathbb{R}^{V \times D}$ 中**取行**。约定 padding 位置（`padding_idx`）的输出强制为全零向量。

$$ \text{out}[t] = \begin{cases} \mathbf{0} & \text{indices}[t] = \text{padding\_idx} \\ W[\text{indices}[t]] & \text{否则} \end{cases} $$

## 规则

- 实现 `solve(weight, indices, padding_idx=None)`，`weight` 形状 `(V, D)`，`indices` 为整数张量（任意形状），输出形状 `(*indices.shape, D)`。
- 无 `padding_idx` 时等价于按行取值 `weight[indices]`。
- 给定 `padding_idx` 时，索引等于它的位置**整行输出置零**（这是本关的前向约定；注意 `F.embedding` 的 `padding_idx` 只影响初始化与梯度，前向并不会置零，故本题按此约定手改）。
- 输出 dtype 跟随 `weight`。

## 示例

```
输入：weight=arange(12).reshape(4,3), indices=[0,3,1,3], padding_idx=3
输出：[[0,1,2], [0,0,0], [3,4,5], [0,0,0]]   # 索引 3 的两处被置零
```

## 约束

- `indices` 取值在 $[0, V)$；float32/float64 权重
