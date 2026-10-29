# 矩阵范数 matrix_norm（fro/nuc/诱导 1·2·inf 范数）

## 题目描述

矩阵范数有两条完全不同的血统：**元素式**的 Frobenius / 核范数（直接把元素平方或把奇异值相加）与**诱导式**的算子范数（$\max_{\|v\|=1}\|Av\|$ 化简后的行/列极值）。`torch.linalg.matrix_norm` 用同一个 `ord` 参数管两头，`1` 是**最大列和**、`inf` 是**最大行和**，方向与向量范数的「绝对值和」毫无关系。谱范数 $\sigma_{\max}$ 与核范数 $\sum\sigma_i$ 则是奇异值上的 max 与 sum，出现在权重正则、谱归一化（GAN）与低秩近似里。

$$\|A\|_{\mathrm{fro}} = \Big(\sum_{ij} a_{ij}^2\Big)^{1/2}, \qquad \|A\|_1 = \max_j \sum_i |a_{ij}|, \qquad \|A\|_\infty = \max_i \sum_j |a_{ij}|, \qquad \|A\|_2 = \sigma_{\max}(A), \qquad \|A\|_{\mathrm{nuc}} = \sum_i \sigma_i(A)$$

## 规则

- 完整签名与参数含义只写在 starter 的 docstring 里，那是唯一判题口径。
- `dim` **恰好两个 int**（行轴、列轴），默认 `(-2, -1)`；其余轴一律当 batch，输入至少二维。
- `ord` 只接受 `"fro"`（默认）、`"nuc"`、`1`、`-1`、`"inf"` / `float("inf")`、`"-inf"`、`2`、`-2`，其余抛 ValueError。
- 负号档位取**另一侧极值**：`-1` 是最小列和、`-inf` 是最小行和、`-2` 是最小奇异值。
- `keepdim=True` 时两根被归约的轴都保留为长度 1，**位置仍在原轴上**（batch 形状不变）。

## 示例

$M=\begin{pmatrix}1&-2&3\\4&0&-1\end{pmatrix}$，八档 `ord` 的标量结果（数值取自 test_cases.py 回填的 EXAMPLE_* 常量）：

```
M = [[ 1, -2,  3],
     [ 4,  0, -1]]

fro  = 5.5677642822265625    # sqrt(31)
nuc  = 7.860636711120605
1    = 5.0                   # 列绝对值和 (5, 2, 4) 取 max
-1   = 2.0                   # 同一组取 min
inf  = 6.0                   # 行绝对值和 (6, 5) 取 max
-inf = 5.0                   # 同一组取 min
2    = 4.159660339355469     # 最大奇异值
-2   = 3.7009761333465576    # 最小奇异值
```

列和那一组是 $\sum_i|M_{i0}|=5$、$\sum_i|M_{i1}|=2$、$\sum_i|M_{i2}|=4$，所以 `ord=1` 给 5、`ord=-1` 给 2；行和是 6 与 5，所以 `ord=inf` 给 6。

## 约束

- 输入 float32（可含 batch 维），内部 float64 计算，返回 float32；与 `torch.linalg.matrix_norm` 对拍 atol=1e-5、rtol=1e-5
- 奇异值非负且降序，因此 2/-2/nuc 三档对拍稳定；**不断言奇异向量**（符号不定）
- `dim` 非二元、越界、两轴相同，或 `ord` 不在名册内，都须抛 ValueError
