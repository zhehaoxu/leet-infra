# 量化：ShiftAddLLM 位平面标度的 ApOT 移位分解

## 题目描述

乘法是最贵的算子。ShiftAddLLM（Park 等, arXiv:2406.05981, NeurIPS 2024）走的是"无乘法重参数化"这条路：权重矩阵先按 BCQ 的口径分解成**位平面**——每平面一个符号矩阵乘一个标量——再把每个标量压成**少数几个 2 的幂的代数和**（ApOT, additive powers of two）。合起来，一次矩阵乘就变成"按符号选平面、移位、相加"。

本题考这条链上确定性的那一段：展平分组 → 贪心位平面 → 对每平面标度做 ApOT 分解 → 用分解后的标度重建。上游的 15 轮 CG/BST 精化、CUDA kernel 与 `q_bias`/`scale` 分支都不在范围内。难点不在数学，而在上游代码里四个容易写错的口径：`torch.round` 是**半偶舍入**而不是 floor、`log(0) = -inf` 配上 `sign(0) = 0` 让零项停在 0、`view([-1, group_size])` 是**行主序展平**后切组所以组会跨行、重建吃的是 ApOT 之后的标度而不是原始标度。

## 规则

记权重为 $W\in\mathbb{R}^{R\times C}$，$n$ 为组数。内部一律 float64，返回时按输入 dtype 给出。

1. **分组**：$g=C$（当 `group_size <= 0`，包括 `-1`）否则 $g=\texttt{group\_size}$；把 $W$ **按行主序展平**成 $R\cdot C$ 个元素后切成 $n=R\cdot C/g$ 组，每组 $g$ 个元素。要求 $g$ 整除 $R\cdot C$；$g$ 不整除 $C$ 时组会跨过行边界，这是上游行为，不是错误。
2. **贪心位平面**（`greedy_mean_torch`，权重掩码 `wf` 为全 1）：$r_0=W_{\text{组}}$，对 $i=0..B-1$（$B=\texttt{num\_bits}$）
   $$b_i=\operatorname{sign}(r_i),\qquad \alpha_i=\frac{\sum_j |r_{i,j}|}{g},\qquad r_{i+1}=r_i-b_i\,\alpha_i .$$
   `planes[:, :, i] = b_i`、`alpha[:, i] = α_i`。$b_i$ 取值 $\{+1,-1,0\}$；某个元素一旦落到 0，$\operatorname{sign}(0)=0$ 会让它之后所有平面都是 0。
3. **ApOT 分解**（`additive_power_of_2`，对每个 $\alpha$ 独立做，$S=\texttt{apot\_nums}$）：
   $$t_0=2^{\operatorname{round}(\log_2|\alpha|)}\cdot\operatorname{sign}(\alpha),\qquad
     t_k=2^{\operatorname{round}(\log_2|d_k|)}\cdot\operatorname{sign}(d_k),\quad d_k=\alpha-\sum_{m<k}t_m .$$
   这里的 $\operatorname{round}$ 是 `torch.round`（就近取整、恰好并列时取偶），不是 floor：$\log_2 3=1.585\to 2$、$\log_2 0.4=-1.322\to -1$。$\alpha=0$ 或残差恰为 0 时 $\log 0=-\infty$、$2^{-\infty}=0$、$\operatorname{sign}(0)=0$，于是 $t_k=0$。`alpha_apot` $=\sum_{k<S}t_k$，`apot_terms` 按提交顺序保存 $t_0..t_{S-1}$。
4. **重建**：$\hat W_{g,j}=\sum_i \texttt{planes}[g,j,i]\cdot\texttt{alpha\_apot}[g,i]$（上游的 `einsum('ijl,il->ij')`），按行主序还原成 $(R,C)$；$\texttt{mse}=\operatorname{mean}\bigl((W-\hat W)^2\bigr)$ 对全部 $R\cdot C$ 个元素取均值。

返回一个字典，键固定为 7 个：

- `group_shape` tuple[int]：$(n, g)$；
- `planes` float $(n, g, B)$：元素只取 $-1/0/+1$，第 2 轴是位平面下标；
- `alpha` float $(n, B)$：贪心位平面的原始标度；
- `alpha_apot` float $(n, B)$：ApOT 分解后的标度；
- `apot_terms` float $(n, B, S)$：每个标度的各个 2 的幂项，按提交顺序；
- `recon` float $(R, C)$：用 `alpha_apot` 重建出的权重；
- `mse` float 0 维：$W$ 与 `recon` 的 MSE。

## 任务

- 输入：`weight` 为 float32 或 float64 的 $(R,C)$ 张量（非连续张量按行主序读取）；`num_bits`、`apot_nums` 为 $\ge 1$ 的 int；`group_size` 为 int，$\le 0$ 时退化成整行一组。
- 输出：上述 7 键字典；浮点容差 float32 用 `1e-5`、float64 用 `1e-9`；`group_shape` 按 Python 元组比较。
- 下列非法输入抛 `ValueError`：`weight` 不是 2 维 / 元素数为 0 / dtype 不是 float32 或 float64、`num_bits < 1`、`apot_nums < 1`、三个整数参数里出现非 int（bool 也算非法）、`group_size > 0` 但不整除 $R\cdot C$。

## 示例

示例一：$2\times 4$ 的权重、`group_size=-1`（整行一组）、$B=2$、$S=2$。第 0 行第 0 个元素 $1.0$：$\alpha_0=\operatorname{mean}(|W|)=2.5$、$b_0=+1$，残差 $1.0-2.5=-1.5$ 故 $b_1=-1$；而 $2.5$ 拆成 $2+0.5$、$1.5$ 拆成 $2-0.5$，于是 $\hat w=1\cdot 2.5+(-1)\cdot 1.5=1.0$。第 1 行的第 3 个元素是 0，两个平面都停在 0。

```
weight     = [[1.0, 1.0, -3.0, 5.0], [2.0, -1.0, 0.0, 1.0]]
num_bits   = 2
group_size = -1
apot_nums  = 2

group_shape   = (2, 4)
planes[0][0]  = [1, -1]
planes[0][1]  = [1, -1]
planes[0][2]  = [-1, -1]
planes[0][3]  = [1, 1]
planes[1][0]  = [1, 1]
planes[1][1]  = [-1, 0]
planes[1][2]  = [0, 0]
planes[1][3]  = [1, 0]
alpha         = [[2.50000000, 1.50000000], [1.00000000, 0.25000000]]
alpha_apot    = [[2.50000000, 1.50000000], [1.00000000, 0.25000000]]
apot_terms    = [[[2.00000000, 0.50000000], [2.00000000, -0.50000000]],
                 [[1.00000000, 0.00000000], [0.25000000, 0.00000000]]]
recon         = [[1.00000000, 1.00000000, -4.00000000, 4.00000000],
                 [1.25000000, -1.00000000, 0.00000000, 1.00000000]]
mse           = 0.32031250
```

示例二：$2\times 3$ 的权重、`group_size=2`。$g=2$ 不整除 $C=3$，展平后是 $[1,2,4,8,16,32]$，切成 3 组 $[1,2]$、$[4,8]$、$[16,32]$ —— 第 2 组横跨了两行。$\alpha=[1.5,6,24]$，最近幂分别是 $2,8,32$（$6\to 2^{\operatorname{round}(2.585)}=8$，$24\to 2^{\operatorname{round}(4.585)}=32$），于是同一行的第 2、3 列拿到了不同的标度。

```
weight     = [[1.0, 2.0, 4.0], [8.0, 16.0, 32.0]]
num_bits   = 1
group_size = 2
apot_nums  = 1

group_shape   = (3, 2)
planes[0][0]  = [1]
planes[0][1]  = [1]
planes[1][0]  = [1]
planes[1][1]  = [1]
planes[2][0]  = [1]
planes[2][1]  = [1]
alpha         = [[1.50000000], [6.00000000], [24.00000000]]
alpha_apot    = [[2.00000000], [8.00000000], [32.00000000]]
apot_terms    = [[[2.00000000]], [[8.00000000]], [[32.00000000]]]
recon         = [[2.00000000, 2.00000000, 8.00000000],
                 [8.00000000, 32.00000000, 32.00000000]]
mse           = 45.50000000
```

## 约束

- `planes` 的第 1 轴是**展平后的组内位置**，不是行号；只有 $g$ 整除 $C$ 时 `planes[k]` 才正好对应 $W$ 的第 $k$ 行。
- ApOT 的第 $k$ 项吃的是"原标度减去已提交各项之和"的**残差**，不是原标度本身；每一项可正可负，所以 $t_0$ 可以大于 $\alpha$。
- `alpha` 与 `alpha_apot` 是两个不同的量，`recon`/`mse` 只用后者；$S=1$ 时 `alpha_apot` 就是 $\alpha$ 的最近幂，$S$ 增大只会让 `alpha_apot` 更贴近 `alpha`，但不会改变 `planes`。
- 取整是"就近取整"而不是 floor：$\alpha=3$ 时 $t_0=4$（floor 会给出 2），$\alpha=0.4$ 时 $t_0=0.5$（floor 会给出 0.25）。并列点只可能出现在 $\alpha=2^{k+1/2}$ 上，那是无理数，二进制输入基本踩不到，判题也不依赖它。
- 判题只看返回字典，不限定实现方式。
