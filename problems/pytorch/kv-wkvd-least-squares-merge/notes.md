# 解题说明 · WKVD 最小二乘合并

## 一句话

被驱逐的 value 不是丢掉，而是**投影到存活 value 张成的子空间里，再按投影系数摊回存活行**。

## 维度账必须先算清

这是本题最容易写反的地方。目标函数是

\[
\min_{C}\ \big\|V_{\text{keep}}^{\top}C - V_{\text{evict}}^{\top}\big\|_F,
\quad V_{\text{keep}}^{\top}\in\mathbb R^{d\times K},\
V_{\text{evict}}^{\top}\in\mathbb R^{d\times E}
\]

所以要解的线性方程组是 \(d\) 个方程、\(K\) 个未知数（每一列 \(C_{:,e}\) 独立），解出来 \(C\in\mathbb R^{K\times E}\)。另一种写法 \(V_{\text{keep}}W\approx V_{\text{evict}}\) 的维度账是 \((K\times d)W = E\times d\)，\(W\) 得是 \(d\times d\) 且只在 \(K=E\) 时有意义——不是同一个问题，也解释不了「每个存活行吸收一份被驱逐分量」。

合并之所以是 \(V_{\text{keep}} + C V_{\text{evict}}\)，看单行就很清楚：

\[
V'_i = V_{\text{keep},i} + \sum_{e=1}^{E} C_{i,e}\, V_{\text{evict},e}
\]

第 \(i\) 个存活 token 吸收的是**所有**被驱逐向量的一个加权和，权重就是它在第 \(i\) 行的拟合系数。这也是为什么它叫「合并」而不是「丢弃」。

## 为什么必须用伪逆

正规方程 \(V_{\text{keep}}V_{\text{keep}}^\top C = V_{\text{keep}}V_{\text{evict}}^\top\) 的系数矩阵是 \(K\times K\) 的 Gram 矩阵。低秩 head 上 \(V_{\text{keep}}\) 的行经常线性相关（\(K > \operatorname{rank}\)），Gram 矩阵奇异，`inv` 直接崩、`solve` 给出 inf/nan。

Moore–Penrose 伪逆给的是**最小范数最小二乘解**：

\[
C^\* = V_{\text{keep}}^{+\top} V_{\text{evict}}^\top
\]

它在秩亏时是所有最优解里 \(\|C\|_F\) 最小的那个——工程意义是「用尽量小的改动吸收被驱逐信息」，不会因为基向量重复而把同一份信息加两遍。实现上 `torch.linalg.pinv` 内部走 SVD，把小于阈值的奇异值当 0 处理，所以退化情形也返回有限值。

## 三个可判的物理性质

1. **满基零残差**：\(K \ge \operatorname{rank}(V_{\text{evict}}^\top)\) 时被驱逐向量完全可表达，残差 0，信息「无损」。
2. **正交分量为 0 系数**：把 \(V_{\text{evict}}\) 换成正交补方向的向量，拟合系数全是 0，`merged == V_keep`，残差正好等于该向量的范数。这条把「投影」语义钉死——如果实现里偷偷加了 bias 或者对 \(V_{\text{evict}}\) 归一化，这一条立刻挂。
3. **双线性**：\(V_{\text{evict}}\to 2V_{\text{evict}}\) 时 \(C\to 2C\)（线性最小二乘解的齐次性），而增量 \(C V_{\text{evict}}\to 4C V_{\text{evict}}\)。很多人会理所当然地以为增量也翻倍，这是最常见的错。

## 边界

- `K`/`E` 为 0、`d` 不一致、非二维输入 → `ValueError`；不要「容错」成空张量，那会让下游的合并静默变成 no-op。
- 内部全程 float64：float32 下对近奇异矩阵做 SVD，小奇异值会被噪声放大，系数与残差的偏差能到 \(10^{-2}\) 量级。
- 残差是**求解残差**而**不是**合并后的重建误差；\(V'\) 与 \(V_{\text{evict}}\) 维度都不同，谈不上距离。

## 复杂度

`pinv` 走 SVD：\(O(dK^2)\)（\(d\ge K\) 时），矩阵乘 \(O(dKE)\)。相比「多留几条 KV」换来的显存收益，这个代价只在压缩触发时付一次，不进解码热路径。

## 相关题目

- `kv-cake-bidirectional-budget`：决定**哪几层砍多少**，是预算分配视角；本题处理「砍掉的内容怎么不浪费」。
- `kv-layer-budget-knapsack`：层间预算的离散优化；同样是资源分配，但目标是最大化保留的注意力质量。
- `kv-h2o-heavy-hitters`：按累计注意力选重击手，选完直接丢其余；本题展示「丢」之前还能先合并。
