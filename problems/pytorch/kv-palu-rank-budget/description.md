# KV Cache：Palu 分组秩预算分配

## 题目描述

Palu（2407.21118）压的是 KV 投影矩阵本身：把 `k_proj`/`v_proj` 的权重按 head 分组，**组内所有 head 共用一组低秩基**，该组的秩 \(r_i\) 就是 cache 里真正存下来的 latent 维度（上游把 latent 拼成 `sum(ranks)` 维，见 `svd_linear.py::HeadwiseLowRankModule`）。整层的保留比例是 \(\sum_i r_i / (G\cdot \texttt{lr\_group\_dims})\)——秩怎么分，直接决定存多少字节。

各组敏感度差得很远，均分秩会在不重要的组上浪费名额。上游 `rank_search.py` 的 fisher 分支因此按 Fisher 信息**比例切分总预算**：

\[
G=\frac{\text{rows}}{\texttt{head\_group\_size}},\qquad
f_i=\operatorname{mean}\bigl(\text{fisher}[\text{第 } i \text{ 组的整块}]\bigr)
\]

\[
r^{\text{float}}_i=\frac{T\,f_i}{\sum_j f_j},\qquad
T=\texttt{ratio}\cdot \texttt{total\_rank},\qquad
\texttt{total\_rank}=G\cdot \texttt{lr\_group\_dims}
\]

小数秩不能拿去建基，于是再走三步：封顶的 floor、按**余数升序**轮转补 1、以及 \(\max(1,\operatorname{round}(r/b))\cdot b\) 的块对齐。本题就是这段**预算账**——SVD、whitening、Hadamard 旋转和 latent 量化都不考。

三个容易写错的细节：

1. 领奖顺序是余数**升序**，不是 largest-remainder：上游 `sorted(..., key=rank_float - floor)` 字面就这么写；
2. 秩发到 \(\texttt{lr\_group\_dims}\) 就停：满组被跳过，指针继续往后走，所以补 1 可能横跨很多轮；
3. 块对齐后**不再截回** \(\texttt{lr\_group\_dims}\)，也不管 `need` 有没有用完，所以 `rank_sum` 可能高于也可能低于 \(T\)。

## 任务

实现 `solve(fisher, param_ratio_target, head_group_size, lr_group_dims, block_size=32)`：

- `fisher`：`(G*head_group_size, lr_group_dims)` 浮点张量，逐元素是低秩基的敏感度权重；
- `param_ratio_target`：\(\rho\in(0,1]\)；`head_group_size`、`lr_group_dims`、`block_size`：正整数（`bool` 视为非法）。

内部一律 float64，浮点输出回落到输入 dtype；不得写回 `fisher`。返回 6 键字典：

- `rank_float`：`(G,)`，输入 dtype，\(T f_i / \sum_j f_j\)；
- `rank_floor`：`(G,)` int64，\(\min(\texttt{lr\_group\_dims},\ \lfloor r^{\text{float}}_i\rfloor)\)；
- `rank_topup`：`(G,)` int64，在 `rank_floor` 上按余数升序轮转发 `need` 个 1，`need = floor(T - Σ rank_floor)`；
- `rank_aligned`：`(G,)` int64，\(\max\bigl(1,\ \operatorname{round}_{\text{half-even}}(\texttt{rank\_topup}_i / \texttt{block\_size})\bigr)\cdot \texttt{block\_size}\)；
- `rank_sum`：0 维 int64，\(\sum_i \texttt{rank\_aligned}_i\)；
- `stored_ratio`：0 维，输入 dtype，`rank_sum / total_rank`。

非法输入抛 `ValueError`：`fisher` 不是二维张量、不是浮点张量、行数不能被 `head_group_size` 整除、列数不等于 `lr_group_dims`、各组 Fisher 之和不为正；`param_ratio_target` 不在 \((0,1]\)；三个整数参数不是正整数。

## 示例

```
fisher          # (16, 32)：4 组 × 每组 4 行，四组的值分别是 4.0 / 3.0 / 2.0 / 1.0
head_group_size = 4        # G = 16 // 4 = 4
lr_group_dims   = 32       # total_rank = 4 * 32 = 128
param_ratio_target = 0.5   # T = 128 * 0.5 = 64.0
block_size      = 8

f            = [4.0, 3.0, 2.0, 1.0]          # 每组整块 (4, 32) 取均值，sum_f = 10.0
rank_float   = [25.6, 19.2, 12.8, 6.4]       # 64.0 * f / 10.0
rank_floor   = [25, 19, 12, 6]               # Σ = 62，余量 T - Σ = 2.0
remainder    = [0.6, 0.2, 0.8, 0.4]          # 升序 → 领奖顺序 [1, 3, 0, 2]
rank_topup   = [25, 20, 12, 7]               # need = floor(2.0) = 2：组 1、组 3 各 +1
rank_aligned = [24, 16, 16, 8]               # round(3.125)=3、round(2.5)=2、round(1.5)=2、round(0.875)=1
rank_sum     = 64                            # 24 + 16 + 16 + 8
stored_ratio = 0.5                           # 64 / 128
```

两个 1 名额发给了余数最小的组 1 和组 3，而不是余数最大的组 2 和组 0——这跟教科书式的最大余额法正好相反，是上游代码的字面行为。`stored_ratio` 落回 0.5 也只是巧合：块对齐把 12 抬成 16、把 25 压成 24，精心算出的比例全被抹平；测试里 `ratio = 0.5` 会对齐出 `rank_sum = 20`（高于 \(T=12\)），`ratio = 0.55` 又对齐出 64（低于 \(T=70.4\)），两个方向都能偏。

## 约束

- **组 Fisher 是整块均值**：对每组 `(head_group_size, lr_group_dims)` 的**整个块**取 mean，不是行均值、也不是先按列再按行。示例里每行取值相同，两种口径凑巧相等，换输入就不行了。
- **余数升序领奖**：排序键是 `rank_float - rank_floor`，用的是**封顶之后**的 `rank_floor`；升序，同余数取更小的组号（稳定排序）。
- **封顶两处**：`rank_floor` 被 `min(..., lr_group_dims)` 截住；轮转时已经等于 `lr_group_dims` 的组直接跳过、指针继续往后走，所以 `need` 可能跨多轮才发完。
- **`need = floor(T - Σ rank_floor)`**：不足 1 的小数余量不发出去，`check_fractional_budget` 把补完的总量钉在 70 而不是 71。
- **块对齐：half-to-even、保底一块、不封顶**：`20/8 = 2.5` 与 `12/8 = 1.5` 都归到偶数 2 → 16，`rank_topup = 0` 的组也被抬成 `block_size` 一块；对齐后不再截回，故 `rank_aligned` 可以大于 `lr_group_dims`（当它不是 `block_size` 的倍数），`stored_ratio` 因此不等于 `param_ratio_target`。
- **dtype 与形状**：`rank_float`/`stored_ratio` 用输入 dtype，其余四键 int64，`rank_sum`、`stored_ratio` 是 0 维张量；`fisher` 必须原样返回。
