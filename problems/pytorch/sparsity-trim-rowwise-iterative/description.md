# 稀疏化：TRIM 逐行稀疏率余弦反馈迭代

## 题目描述

统一的逐行稀疏率 \(s\) 默认每一行都剪掉同样比例的列，但一层的输出行对「被删元素」的敏感程度并不一样：同样剪掉 \(s\) 比例的列，有的行输出几乎不变，有的行方向已经偏了。TRIM（*Achieving Extreme Sparsity with Targeted Row-wise Iterative Metric-driven Pruning*, 2505.16743）因此给**每个输出行一个不同的稀疏率**，并用「剪枝前后输出行的余弦相似度」当反馈信号迭代调整。

### 1. 排序由 Wanda metric 给出

先算逐元素分数

\[
m_{ij}=|W_{ij}|\cdot\sqrt{\mathbb{E}_t[X_{t,j}^2]}
\]

对每一行做**稳定升序**排序，得到列的处理顺序 `sorted_idx`；之后无论该行稀疏率是多少，剪掉的永远是这份顺序里**前 `int(in_features * s_i)` 个**列（同分取更小的列下标）。

### 2. 迭代调整逐行稀疏率

记 \(Y=WX\) 为未剪枝输出，\(\hat Y(x)\) 为按稀疏率向量 \(x\) 剪枝后的输出。初始 \(x_i=s\)。每一轮：

\[
q_i=\cos\bigl(Y_i,\hat Y_i\bigr),\quad
d_i=\frac{q_i-\min q}{\max q-\min q+10^{-6}},\quad
x\leftarrow x+2\eta\left(d-\overline d\right)
\]

然后做截断与重新对齐：

\[
x\leftarrow \mathrm{clip}(x,0,0.95),\qquad
x\leftarrow x-\overline x+s
\]

第二步保证整层的**平均稀疏率仍然等于 \(s\)**——迭代只重新分配行间预算。每轮更新后用新的 \(x\) 重新剪枝、重算输出与整层扁平余弦，只保留扁平余弦最大的那组 \(x\)（`best_x`）及其轮次（`best_iter`，0 起算）。

### 3. 学习率网格

\(\eta\) 不在迭代里自适应，而是外层扫描：

- 正网格 `[0.01, 0.02, 0.04, 0.08, 0.12, 0.16]`：按顺序尝试，若某次得分不高于当前最好得分就**立刻停止**；
- 只有「正网格从未超过初始基线」且未设置 `forbid_neg_lr` 时，才尝试负网格 `[-0.01, -0.02, -0.04]`；
- 只有得分**严格超过**当前最优时，才更新最优 \(x\)、`learning_rate` 与 `best_iter`。因此若所有学习率都没能超过基线，`learning_rate` 就是 `None`。

### 4. 后处理与掩码

拿到最优 \(x\) 后，上游还会再做一次 `>0.95→0.95`、`<0→0`、再减均值加 \(s\)，然后才按 `int(in_features * s_i)` 生成掩码。注意**后处理在择优之后**：返回的 `best_metric` / `learning_rate` 对应的是后处理前的 \(x\)。

## 任务

实现 `solve(weight, inputs, activation_ms, target_sparsity, iterations=10, qmetric="cosim_flat", qmetric_dimwise="cosim", forbid_neg_lr=False, lr_grid=(0.01,...,0.16), neg_lr_grid=(-0.01,-0.02,-0.04))`：

- `weight`：`(out, in)`，float32/float64，`out >= 2`；
- `inputs`：`(in, tokens)`，同 dtype，`tokens >= 1`；
- `activation_ms`：`(in,)`，同 dtype，排序用的逐列均方；
- `target_sparsity`：`(0, 1)`；`iterations >= 0`；
- 只支持 `qmetric="cosim_flat"`、`qmetric_dimwise="cosim"`，其余值抛 `ValueError`。

内部 float64。余弦都带 `1e-8` 的范数保护并夹到 `[-1, 1]`；`flat` 指整层展平，`dimwise` 指逐输出行；`starting_var`/`ending_var` 是逐行余弦（**无偏**方差，`ddof=1`）在初始/最优稀疏率下的值。

返回 9 键字典：

- `sparsities`：`(out,)` float64，**后处理后**的逐行稀疏率；
- `keep_mask`：`(out, in)` bool，True 表示保留；
- `pruned_weight`：`(out, in)`，按输入 dtype；
- `baseline_metric`、`best_metric`、`starting_var`、`ending_var`：python float；
- `learning_rate`：写入选中的学习率，可能为 `None`；
- `best_iter`：python int。

## 示例

```
weight = [[3,2,1,0],[0,1,2,3],[1,1,1,1],[4,0,0,1]]
inputs = [[1,0,1],[0.5,1,-1],[2,0.5,1],[1,1.5,-1]]   # (in=4, tokens=3)
activation_ms = [2/3, 0.75, 1.75, 17/12]
target_sparsity = 0.5, iterations = 5
```

排序后每行的剪枝顺序由 metric 决定；迭代 5 轮、扫描正学习率网格，`0.01` 一次就超过了基线：

```
baseline_metric = 0.9856996457     best_metric = 0.9928880902
learning_rate   = 0.01             best_iter   = 0
starting_var    = 8.6939e-06       ending_var  = 1.2014e-18
sparsities      = [0.48964132, 0.49108361, 0.50963753, 0.50963754]
keep_mask       = [[T,T,T,F],
                   [F,T,T,T],
                   [F,F,T,T],
                   [T,F,F,T]]
pruned_weight   = [[3,2,1,0],
                   [0,1,2,3],
                   [0,0,1,1],
                   [4,0,0,1]]
```

四个稀疏率的均值正好是 `0.5`：迭代只把预算从「不敏感的行」挪给「敏感的行」。行 2 的 `0.5096` 比行 0 的 `0.4896` 更狠，说明它在这一轮里被认为更容易剪。

## 约束

- **排序只做一次**：每轮只换稀疏率，不重排 metric，因此被剪集合始终是每行 `sorted_idx` 的前缀。
- **平均预算守恒**：迭代内与后处理各做一次「减均值加 target」，最终均值等于 `target_sparsity`（在 `0.95`/`0` 截断未改变均值的前提下；若截断触发，均值会被重新拉回 target）。
- **择优是严格大于**：得分相等不更新最优；`learning_rate` 可能为 `None`。
- **停止规则**：某个 \(\eta\) 的得分不高于「本网格当前最好」就停止扫描，不再尝试更大的 \(\eta\)；负网格只在正网格一无所获时启用。
- **方差口径**：`torch.var` 默认无偏（`ddof=1`），`out=1` 会得到 NaN，所以 `out >= 2`。
- **后处理在择优之后**：不要用后处理后的稀疏率回填 `best_metric`/`learning_rate`。
