# KV Cache：WKVD 值向量的最小二乘合并

## 题目描述

KV cache 压缩到了必须丢 token 的那一步，常见的做法是「丢最近的」或「丢分数最低的」，但直接丢掉一条 value 向量等于把这条 key 上的信息彻底抹掉。WKVD 一类方法的观察是：**被驱逐的 value 向量往往能被存活 value 的线性组合近似重建**——attention head 的 value 空间是低秩的，冗余度很高。既然能重建，就不必真丢，把它**投影到存活张成的子空间里合并进去**即可。

形式化：存活 value 矩阵 \(V_{\text{keep}}\in\mathbb R^{K\times d}\)（每行一个 token 的 value），被驱逐的 \(V_{\text{evict}}\in\mathbb R^{E\times d}\)。我们要找系数矩阵 \(C\in\mathbb R^{K\times E}\) 使得

\[
C^\*=\arg\min_{C}\ \big\|V_{\text{keep}}^{\top}C - V_{\text{evict}}^{\top}\big\|_F
\]

然后把被驱逐的信息加进存活行：

\[
V' = V_{\text{keep}} + C\,V_{\text{evict}} \in \mathbb R^{K\times d},
\qquad \text{residual} = \big\|V_{\text{keep}}^{\top}C - V_{\text{evict}}^{\top}\big\|_F
\]

注意目标里是**列方向**的最小二乘：\(V_{\text{keep}}^\top\) 是 \((d\times K)\)，\(V_{\text{evict}}^\top\) 是 \((d\times E)\)，所以 \(C\) 有 \(K\times E\) 个未知数，每一列独立地解一个「用 \(K\) 个基向量拟合一条被驱逐向量」的问题。

反过来看 \(V'\) 的每一行就很直观：\(V'_i = V_{\text{keep},i} + \sum_{e} C_{i,e}\,V_{\text{evict},e}\)——第 \(i\) 个存活 token 吸收了一部分被驱逐 token 的分量。

三个性质决定了这道题的判分点：

1. **满基必然零残差**。若 \(K\ge\) 张成维数，被驱逐向量都能被精确表达，残差为 0，合并是「无损」的。
2. **正交方向无法表达**。分量落在 \(V_{\text{keep}}\) 行空间的正交补里时，系数必为 0，合并结果与原来一致，残差等于该分量的范数。
3. **线性相关必须用伪逆**。\(V_{\text{keep}}\) 的行可能线性相关（这在低秩 head 上很常见），此时 \(V_{\text{keep}}^\top\) 不满列秩，正规方程无唯一解；必须用 Moore–Penrose 伪逆取**最小范数解**，而不是报错或给出爆炸值。

## 任务

实现 `solve(V_keep, V_evict)`：

- 输入：`V_keep`（float32 `(K, d)`）、`V_evict`（float32 `(E, d)`），\(K,E\ge 1\)。
- 输出 `(merged, coef, residual)`：`merged` 为 `(K, d)` float32；`coef` 为 `(K, E)` float32；`residual` 为 0 维 float32 标量（Frobenius 残差 \(\|\cdot\|_F\)）。
- 内部用 float64 求解，输出转回 float32。
- `d` 不一致、或 `K`/`E` 为 0、或输入不是二维，抛 `ValueError`。
- 不得对 \(V_{\text{keep}}\) 加正则、不得要求满秩——退化情形（线性相关行、\(K>d\)）必须照常返回有限值。

## 示例

```
V_keep  = [[1, 0],
           [0, 1]]            # 标准正交基，K = d = 2
V_evict = [[0.5, 0.25]]       # E = 1

basis  = V_keep^T = [[1,0],[0,1]]
target = V_evict^T = [[0.5],[0.25]]
coef   = pinv(basis) @ target = [[0.5],[0.25]]      # 满基，直接读出分量

merged = V_keep + coef @ V_evict
       = [[1, 0], [0, 1]] + [[0.5],[0.25]] @ [[0.5, 0.25]]
       = [[1, 0], [0, 1]] + [[0.25, 0.125],
                             [0.125, 0.0625]]
       = [[1.25,  0.125],
          [0.125, 1.0625]]

residual = ||basis @ coef - target||_F = 0
```

被驱逐的 `[0.5, 0.25]` 被拆成 `0.5×row0 + 0.25×row1`，这两份分别加到两个存活行上（外积形式），所以合并后每个元素都被「摊」了一点。

## 约束

- **最小二乘的方向是列**：解的是 \(V_{\text{keep}}^\top C \approx V_{\text{evict}}^\top\)，不是 \(V_{\text{keep}} W \approx V_{\text{evict}}\)。前者给出 \((K\times E)\) 的系数、合并是加行；后者维度对不上（\(K\times d\) 乘 \(d\times d\) 得不到 \(E\times d\)）。
- **残差是求解残差不是合并残差**：\(\|V_{\text{keep}}^\top C - V_{\text{evict}}^\top\|_F\) 衡量「被驱逐信息能被表达多少」；合并后的 \(V'\) 与 \(V_{\text{evict}}\) 无直接距离可比（维度都不同）。
- **伪逆优先于求逆与 lstsq 报错**：`pinv` 在秩亏时给最小范数解；用 `torch.linalg.inv` 会直接崩，用 `solve` 会得到 inf/nan。
- **合并增量是双线性**：\(V'-V_{\text{keep}} = C\,V_{\text{evict}}\)，把 \(V_{\text{evict}}\) 放大 2 倍时系数也放大 2 倍，增量因此放大 4 倍——不是 2 倍。
- 系数与残差都用 float64 算完再降精度，避免 float32 下 `pinv` 对近奇异矩阵给出噪声。
