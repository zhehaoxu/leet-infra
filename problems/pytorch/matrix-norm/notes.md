# 详解：矩阵范数 matrix_norm（fro/nuc/诱导 1·2·inf 范数）

## 出处

`torch.linalg.matrix_norm` 是 1.9 新线性代数接口的一员，`ord` 名册沿用 numpy 的 `np.linalg.norm` 约定。诱导范数的闭式（列和/行和）在 Horn & Johnson《Matrix Analysis》里是标准结论；谱范数与核范数则分别是 $\sigma$ 向量的 $\ell_\infty$ 与 $\ell_1$。工程侧的落点：Ganin 的梯度反转层谱归一化、Spectral Norm Regularization（Yoshida & Miyato 1705.10921）只压 $\sigma_{\max}$，低秩/蒸馏里用核范数做稀疏代理。

## 诱导 1 / inf 范数为什么是「列和 / 行和」

按定义 $\|A\| \to \max_{\|v\|_p = 1}\|Av\|_p$。取 $p=1$：

$$\max_{\|v\|_1=1}\|Av\|_1 = \max_{\|v\|_1=1}\sum_j\Big|\sum_i a_{ji}v_i\Big| \le \max_i\sum_j|a_{ji}|$$

把 $v$ 取成「全押在最有利的那个输入分量 $i$」的单位向量即可取等，于是 $\|A\|_1 = \max_i \sum_j |a_{ji}|$ = **最大列和**（下标 $i$ 是列）。同理 $p=\infty$ 时 $\|Av\|_\infty \le \max_j\sum_i|a_{ji}|\cdot\|v\|_\infty$，取 $v=\mathrm{sign}(A_{j\cdot})$ 取等，得**最大行和**。这两条正好互为转置：$\|A\|_1 = \|A^{\mathsf T}\|_\infty$——记住这一句就不会写反方向。

参考答案的写法与这个推导逐字对应：

- `ord=1/-1` → `xd.abs().sum(dim=ra, keepdim=True)` 沿**行轴**求和（把行压掉，留下列方向），再沿**列轴** `amax/amin`；
- `ord=inf/-inf` → 沿**列轴**求和，再沿**行轴**取极值。

## fro / nuc / 2 / -2：归约轴与 batch 布局

`"fro"` 就是 $\sqrt{\sum a^2}$，与向量 2-范数同式。$2/-2/\text{nuc}$ 需要奇异值，而 `torch.linalg.svdvals` 要求矩阵轴在**最后两维**，于是：

$$y = \text{movedim}(x; (r, c) \to (n-2, n-1)), \qquad s = \text{svdvals}(y)$$

`svdvals` 返回 $s$ 非负降序，所以 `max`、`min`、`sum` 三个统计量都是确定的数值（不存在奇异向量符号那种歧义）。**坑在布局**：movedim 之后剩下的 batch 轴按原始升序排列，`v` 的形状不再是「与 $x$ 同轴布局、被归约轴长度 1」。参考答案用一个 `_restore` 把结果补回原位：

```
kept = [i for i in range(ndim) if i not in (ra, ca)]
full = [1] * ndim
for i, s in zip(kept, v.shape): full[i] = s   # 按保留维逐轴回填
v = v.reshape(full)
```

若直接 `reshape` 成 `x` 的形状或用 `unsqueeze(-1)`，在 `dim=(0,1)`、`dim=(0,2)` 这类非相邻/非尾部轴选择上会得到错误的轴位置——这是本族实现最常见的 bug（凡「按显式 dims 归约后还要广播回原布局」都该按**保留维**逐项构造 shape）。

## 核心与易错点

1. **`ord=1` 的两个世界**：向量是 $\sum|x|$，矩阵是最大列和。判题把两题放在一起就是逼这个区别。
2. **`dim` 必须是恰好两个 int**：与向量范数允许任意元组不同，`torch.linalg.matrix_norm` 只接受两个轴。
3. **`-2` 是最小奇异值而非「谱范数的相反数」**，非方阵时 $\sigma$ 只有 $\min(m,n)$ 个，`-2` 与 `2` 分别是首尾元素。
4. **bool 不是 int**：`ord=True` 会溜进 `== 1` 分支，参考答案显式排除 bool。
5. **batch 轴在 `dim` 不含它时也要保留**：`x.dim()>2` 且 `keepdim=False` 时输出形状是「除两轴外全部」，不是标量。

## 与邻近题的区别

- `vector-norm`（order358）：同批姊妹题，考向量侧 `ord` 口径与多轴归约。
- `lp-pool-unpool`：$L^p$ 池化是沿单轴的带 $1/n$ 因子的统计量，没有诱导范数概念。
- `rmsnorm` / `layernorm-forward`：仿射归一化前向，不用奇异值。
- `distributed-clip-grad-norm`：把一堆梯度张量展平成一个向量再算全局范数，属向量口径。

## 对拍口径与复杂度

- oracle 是 `torch.linalg.matrix_norm`，覆盖八档 `ord` × 多组 `dim`（相邻、非相邻、含负轴）× `keepdim` 两态，含三维/四维 batch 输入与非法参数探针。
- 复杂度：`fro/1/-1/inf/-inf` 是 $O(mn)$；`2/-2/nuc` 走完整 SVD，$O(mn\min(m,n))$，判题矩阵 $\le 6\times 7$，float64 下仍是毫秒级。
