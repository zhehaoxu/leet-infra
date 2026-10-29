# RoAD：分组二维旋转适配器

## 题目描述

RoAD（Rotation Adaptation）不在权重上加低秩项，而是**把基座层的输出按 2 维一对做旋转**，每个分组里的每一对元素有自己的旋转角与缩放。它不存储 `out_features × out_features` 的旋转阵，只存角度 `theta` 和缩放 `alpha`，推理时用逐元素形式 `y = x * first_col + rotate_half(x) * second_col` 代替矩阵乘法。

本题要求给定角度、缩放、variant、分组大小与输入张量，输出**显式旋转阵 `R`** 与**旋转后的输出 `y`**，并保证两者一致（`y = x @ Rᵀ`）。

## 规则

记 `out_features` 为输出维度，`group_size` 为正偶数，`half = group_size // 2`。把输出切成 `out_features / group_size` 个 group；**每个 group 内第 `j` 个元素与第 `j + half` 个元素配成一对**（不是相邻元素配对）。

1. **参数长度**（`theta` 与 `alpha` 等长）：

   | variant | 长度 | 含义 |
   | --- | --- | --- |
   | `road_1` | `out_features / 2` | 每个 group 有 `group_size/2` 个参数，同一对共用 |
   | `road_2` | `out_features` | 每个位置各一个参数 |
   | `road_4` | `out_features * 2` | 每对两组参数：前半供 `cos` 项，后半供 `sin` 项 |

2. **逐位置的两列系数** `first_col[k]`、`second_col[k]`：

   - `road_1`：先把 `theta`、`alpha` reshape 成 `(n_group, half)`，再对第二维 `repeat_interleave(2)` 展平；然后
     `first_col = alpha * cos(theta)`、`second_col = alpha * sin(theta)`（逐元素）。
   - `road_2`：直接用原始顺序，公式同上。
   - `road_4`：把 `theta` 与 `alpha` reshape 成 `(n_group, 2, group_size)`；
     `first_col = alpha[:, 0, :] * cos(theta[:, 0, :])`、`second_col = alpha[:, 1, :] * sin(theta[:, 1, :])`，各自按 group 展平。

3. **旋转阵**：

   ```
   R = diag(first_col) + 第二列的贡献
   ```

   第二列贡献的构造：把 `second_col` 按 `(n_group, 2, half)` 交换前后两半并展平，取对角；再 reshape 成 `(n_group, 2, half, out_features)`，交换前后两半，并把第一半整体取负；最后 reshape 回 `out_features × out_features`。

   等价的分块写法（每个配对 `a = g*group_size + j`、`b = a + half`）：

   ```
   R[a, a] = first_col[a]
   R[b, b] = first_col[b]
   R[a, b] = -second_col[a]
   R[b, a] =  second_col[b]
   ```

4. **逐元素输出**：把 `x` 按 group 切前后两半 `x1`、`x2`，

   ```
   rotate_half = stack(-x2, x1)         # 与 x 同形
   y = x * first_col + rotate_half * second_col
   ```

   它必须与 `x @ Rᵀ` 相等；当 `alpha` 全为 1 时 `R` 是正交阵，保 2-范数。

`x` 的最后一维必须是 `out_features`，允许任意前导维度。全部中间计算用 float64，返回 float32。

返回 `(R, y)`：`R` 为 `(out_features, out_features)`，`y` 与 `x` 同形。

## 示例

`theta = [0.1, 0.2, 0.3, 0.4]`、`alpha = [1, 1, 1, 1]`、`variant = "road_1"`、`group_size = 4`、`out_features = 8`、`x = [[0, 1, 2, 3, 4, 5, 6, 7]]`

第一个 group 的 4 个位置共用两个角（`0.1` 给第 0/2 位，`0.2` 给第 1/3 位）：

```
R[:4, :4] = [[ 0.9950,  0.0,    -0.0998,  0.0   ],
             [ 0.0,     0.9801,  0.0,    -0.1987],
             [ 0.0998,  0.0,     0.9950,  0.0   ],
             [ 0.0,     0.1987,  0.0,     0.9801]]
```

第二个 group（角 `0.3`、`0.4`）落在 `R[4:, 4:]`，跨 group 的位置恒为 0。

```
y = [[-0.1997, 0.3841, 1.9900, 3.1389, 2.0482, 1.8794, 6.9141, 8.3945]]
```

## 约束

- `group_size` 必须是正偶数，`out_features` 必须能被 `group_size` 整除；
- `variant` 只能是 `road_1` / `road_2` / `road_4`；
- `theta`、`alpha` 必须是一维且长度符合上表；
- `x` 的最后一维必须等于 `out_features`；
- 违反以上任一条时抛出 `ValueError`。
