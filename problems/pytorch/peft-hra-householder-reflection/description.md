# HRA：Householder 反射适配与紧凑 WY 合并

## 题目描述

HRA（Householder Reflection Adaptation）不改基座权重本身，而是在输入侧乘一个正交矩阵 `H`：`H` 是 r 个 Householder 反射的乘积，每个反射由一列可训练向量 `u_k` 决定。合并时把 `H` 乘进基座权重（`W0 @ H`），推理就退化成普通线性层。

难点在于：r 个反射逐个相乘会产生交叉项，不能简单写成 `I - V Vᵀ`。HRA 用紧凑 WY 表示把它压成 `H = I - V T Vᵀ`，其中 `T` 是 r×r 上三角矩阵。

本题要求给定反射向量组 `u`、基座权重 `W0`、是否先做 Gram-Schmidt 正交化，以及前向/反向方向，输出 `V`、`T`、`H` 和合并后的权重。

## 规则

记 `u` 为 `(in_features, r)`，`W0` 为 `(out_features, in_features)`，`rank = r`。全部中间计算在 float64 下进行，返回 float32。

1. **正交化分支**（`apply_gs=True`）：对 `u` 做 reduced QR 分解，取 `V = Q`（`in_features × r`），并令

   ```
   T = 2 * I_r
   ```

   此时 `H = I - V T Vᵀ = I - 2 Q Qᵀ`，即「在 span(u) 的正交补上恒等、在 span(u) 上整体取反」的反射。

2. **非正交化分支**（`apply_gs=False`）：`V` 的每一列是 `u` 对应列的单位化向量，

   ```
   v_k = u_k / ||u_k||
   T_inv = 0.5 * I_r + triu(Vᵀ V, diagonal=1)
   T     = solve_triangular(T_inv, I_r, upper=True)
   ```

   `triu(·, diagonal=1)` 只保留严格上三角。这样得到的 `H = I - V T Vᵀ` 恰好等于按 `P_0 @ P_1 @ … @ P_{r-1}` 顺序逐个相乘的反射乘积（`P_k = I - 2 v_k v_kᵀ`），交叉项被 `T` 还原。

3. **反向**（`reverse=True`）：把 `T` 换成 `Tᵀ`，得到 `Hᵀ`。用于 unmerge。

4. **合并**：`W_merged = W0 @ H`。等价于先对输入做 `x @ Hᵀ` 再过基座线性层。

返回四元组 `(V, T, H, W_merged)`，形状分别为 `(in_features, r)`、`(r, r)`、`(in_features, in_features)`、`(out_features, in_features)`。

## 示例

`u = [[1, 2], [0, 1], [0, 0]]`、`W0 = [[1, 2, 3], [4, 5, 6]]`、`apply_gs = False`

两列不正交（`[(1,0,0)·(2,1,0)]/… ≠ 0`），所以 `T` 有非零的右上元素：

```
V = [[1.0, 0.8944272],
     [0.0, 0.4472136],
     [0.0, 0.0      ]]
T = [[2.0, -3.5777087],
     [0.0,  2.0      ]]
H = [[ 0.6, 0.8, 0.0],
     [-0.8, 0.6, 0.0],
     [ 0.0, 0.0, 1.0]]
W_merged = [[-1.0,  2.0, 3.0],
            [-1.6,  6.2, 6.0]]
```

`H` 是两个反射的乘积，在 3 维里表现为前两维上的旋转（`det H = 1`），第三维不变。

同一个 `u` 改成 `apply_gs = True` 后，`V` 是 `u` 的 QR 正交基（本例就是 `[[1,0],[0,1],[0,0]]`），`T = 2I`：

```
H = [[-1.0,  0.0, 0.0],
     [ 0.0, -1.0, 0.0],
     [ 0.0,  0.0, 1.0]]
W_merged = [[-1.0, -2.0, 3.0],
            [-4.0, -5.0, 6.0]]
```

`reverse=True` 时 `H` 取转置：右上角的 `0.8` 变 `-0.8`，左下角的 `-0.8` 变 `0.8`。

## 约束

- `u` 必须是二维，`r >= 1`，每一列都不能是零向量；
- `W0.shape[1] == u.shape[0]`；
- `apply_gs=True` 时要求 `r <= in_features`；
- 不满足以上条件时抛出 `ValueError`。
