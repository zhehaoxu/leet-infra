# 稀疏化：OWL 离群比驱动的逐层稀疏率分配

## 题目描述

一次性剪枝里，**统一稀疏率**往往不是最优：有的层激活里「离群值」很多，权重与激活的乘积分布被少数大值主导，剪掉尾部更安全；有的层激活平缓，应该少剪。OWL（*Outlier Weighed Layerwise Sparsity*, 2310.05175）的做法是先用一个可观测的统计量给每层打分，再把全局稀疏预算按分数重新分配。

### 1. 每层的 metric 与离群比

对一层里的每个 Linear，先算与 Wanda 相同的 \(m_{ij}=|W_{ij}|\cdot\sqrt{\mathbb{E}_t[X_{t,j}^2]}\)。把**同一层所有 Linear 的 metric 展平后拼在一起**，设整体均值为 \(\mu\)，则离群比定义为

\[
r_l = \frac{\#\{\,m > \mu\cdot \texttt{hyper\_m}\,\}}{\#\{m\}}\times 100
\]

注意是**严格大于**，阈值是「均值 × `hyper_m`」而不是「均值 + k 个标准差」（上游另有 `check_outlier` 用 max 做阈值，但没有被这条路径调用）。

### 2. 归一化与预算对齐

把 \(L\) 层的离群比做 min-max 归一，映射到 \([0, 2\lambda]\)：

\[
\tilde r_l=\frac{r_l-r_{\min}}{r_{\max}-r_{\min}}\cdot 2\lambda
\]

再减均值、加上 \(1-s\)（\(s\) 为全局目标稀疏率）：

\[
d_l=\tilde r_l-\overline{\tilde r}+(1-s),\qquad \text{layer\_sparsity}_l=1-d_l
\]

于是 \(\sum_l \text{layer\_sparsity}_l/L = s\)——归一化只重新分配预算，不改变总量。若某层算出 `layer_sparsity <= 0`（密度超过 1），上游直接把它置成 `0.01`；这只是「1% 稀疏」的硬下限，不是 `1-...`。

### 3. 层内掩码

拿到 `layer_sparsity_l` 后，层内每个 Linear 仍按 Wanda 的方式逐行剪：对本行 metric 做**稳定升序**排序，剪前 `int(in_features * layer_sparsity_l)` 个，同分取更小的列下标。同一层的多个 Linear 共用这个层稀疏率。

## 任务

实现 `solve(weights, activation_ms, sparsity, lamda=0.08, hyper_m=3.0)`：

- `weights`：`[层][Linear]` 嵌套列表，元素是 `(out, in)` 的 float32/float64 张量；
- `activation_ms`：同结构，元素是 `(in,)`，表示该 Linear 输入通道的**均方** \(\mathbb{E}_t[X_{t,j}^2]\)；
- `sparsity`：`[0,1]` 内数值；`lamda > 0`；`hyper_m > 0`；
- 层数必须 `>= 2`，每层至少一个 Linear，所有张量同 dtype。

按上文公式计算离群比、密度与逐层稀疏率；再对每个 Linear 逐行生成掩码。返回 6 键字典：

- `outlier_ratio`、`density`、`layer_sparsity`：`(L,)` float64；
- `metric`、`mask`、`pruned_weight`：与输入同结构的嵌套列表，`metric`/`pruned_weight` 用输入 dtype，`mask` 为 bool 且 True 表示被剪。

**非法输入抛 `ValueError`**：层数不足、空层、结构或 dtype 不一致、激活长度不等于 `in_features`、`sparsity` 越界、`lamda`/`hyper_m` 非正，以及**各层离群比完全相同**——上游此时 `max-min=0`，min-max 归一化会除零，本题把它显式判为非法。

## 示例

三层、每层一个 `1×8` 的 Linear，激活全为 1，`sparsity=0.5`、`lamda=0.08`、`hyper_m=3`：

```
层0 权重 [1,1,1,1,1,1,1,1]      metric 均值 1    阈值 3     离群 0 个 → r = 0
层1 权重 [1,1,1,1,1,1,1,40]     metric 均值 5.875 阈值 17.625 离群 1 个 → r = 12.5
层2 权重 [1,1,1,1,1,1,40,40]    metric 均值 10.75 阈值 32.25 离群 2 个 → r = 25
```

```
r        = [0, 12.5, 25]
归一化    = [0, 0.08, 0.16]          # (r-min)/(max-min) * 2*0.08
减均值     = [-0.08, 0, 0.08]         # 均值 0.08
density  = [0.42, 0.5, 0.58]          # + (1-0.5)
sparsity = [0.58, 0.5, 0.42]          # 1-density
k        = [4, 4, 3]                  # int(8 * sparsity)
```

层 2 的离群最多，获得最低稀疏率 0.42（只剪 3 个）；层 0 没有离群，被剪得最狠（4 个）。掩码分别为 `[T,T,T,T,F,F,F,F]`、`[T,T,T,T,F,F,F,F]`、`[T,T,T,F,F,F,F,F]`。

## 约束

- **离群比按层拼接**：同一层里多个 Linear 的 metric 要拼成一个整体再求均值和阈值，不是每个 Linear 各算各的。
- **严格大于**：恰好等于 `mean*hyper_m` 的元素不算离群，测试用 `[1,1,1,9]` 卡这条。
- **归一化只重分配**：`layer_sparsity` 的均值等于目标 `sparsity`（未被 0.01 下限破坏时）。
- **下限是 `0.01`**：`layer_sparsity <= 0` 时置 `0.01`，此时 `int(in*0.01)` 可能是 0，即该层实际不剪。
- **同分取小下标**：层内仍用稳定升序排序，与 Wanda 题一致。
