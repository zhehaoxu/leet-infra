# 解题说明 · Palu 分组秩预算分配

## 一句话

把 `total_rank * ratio` 这坨预算按各组 Fisher 比例切成整数秩：封顶 floor → 余数**升序**轮转补 1 → `block_size` 块对齐，纯预算账，不做任何分解。

## 账为什么这么算

组敏感度 \(f_i\) 是每组整块的均值，预算 \(T=\rho\cdot G\cdot L\)（\(L=\texttt{lr\_group\_dims}\)），于是

\[
r^{\text{float}}_i = T\,\frac{f_i}{\sum_j f_j},\qquad
r^{\lfloor}_i=\min\bigl(L,\ \lfloor r^{\text{float}}_i\rfloor\bigr)
\]

比例式保证 \(\sum_i r^{\text{float}}_i = T\)：秩跟着敏感度走，而不是跟着头数平摊。floor 之后 shortfall \(d=T-\sum_i r^{\lfloor}_i\) 一般是小数，逐个 1 发回去得到 `rank_topup`；最后 \(\max(1,\operatorname{round}(r/b))\cdot b\) 把 latent 宽度对齐到 \(b\) 的倍数（上游写死 32，让 `sum(ranks)` 维 latent 与量化分组对齐）。三段里只有第三段会破坏总量。

## 四个易错口径

1. **升序领奖**：排序键 `rank_float - rank_floor` 是升序，所以拿到补发的是余数**最小**的组。这跟最大余额法相反，但上游 `sorted(..., key=rank_float - floor)` 字面如此，测试也照抄。
2. **clamp 之后才算余数**：满组 `rank_float` 远大于 \(L\)，clamp 后余数变成 \(r^{\text{float}}_i - L\)，是个很大的正数，于是被推到升序队尾；先算余数再 clamp 会得到完全不同的领奖顺序。
3. **补发次数是 `floor(d)`，满组只跳过不退还**：`need` 用不完时剩下的名额直接丢掉；指针每走一步遇到满组就 `continue`，所以 `need` 可能横跨很多轮才发完。
4. **对齐既不封顶也不守恒**：`max(1, ·)` 给 0 秩组保底一块，`round` 是 half-to-even，算完不再截回 \(L\)；`rank_sum` 因此可以高于也可以低于 \(T\)，`stored_ratio` 不等于 `param_ratio_target`。

## 判分点

- 六键齐全：`rank_float`/`stored_ratio` 跟随输入 dtype（float32 容差 1e-5，float64 容差 2e-9），`rank_floor`/`rank_topup`/`rank_aligned` 是 int64 向量，`rank_sum` 是 0 维 int64；
- `rank_topup` 必须与「升序 + 跳过满组 + 共 `floor(d)` 次」逐元素相等，差一个位置就红；
- `rank_aligned` 同时考 half-to-even（2.5 与 1.5 都归 2）和 0 秩组的保底一块；
- `rank_sum == Σ rank_aligned`、`stored_ratio == rank_sum / total_rank`，两者都跟 `ratio` 没有直接等式关系；
- 内部 float64，且不许写回 `fisher`。

## 边界

- `ratio = 1.0`：\(T=\texttt{total\_rank}\)，各组 floor 撞上 \(L\) 后被跳过，名额全给剩下的组（`check_cap_skips_and_cycles` 里三组最终都满）。
- 某个组 \(f_i=0\) 合法（只要总和大于 0）：`rank_float`、`rank_floor`、`rank_topup` 都是 0，但块对齐仍给它一块。
- 全体为 0 / 总和非正 → `ValueError`；`fisher` 的行列形状不满足整除与列宽 → `ValueError`。
- `need > headroom` 的守卫（`rank budget exceeds available headroom`）在合法输入下打不到：\(\texttt{need}\le T-\sum r^{\lfloor}\le G L-\sum r^{\lfloor}=\texttt{headroom}\)，且 headroom 是整数，floor 之后不会越界；留着只是防 NaN 之类的意外。

## 复杂度

块均值 \(O(G\cdot\texttt{head\_group\_size}\cdot L)\)（即读一遍输入），排序 \(O(G\log G)\)，轮转最坏 \(O(G\cdot\texttt{need})\)——每连续 \(G\) 步至少发掉一个，满组访问只在这 \(G\) 个位置里占坑；对齐与求和 \(O(G)\)。

## 出处

`shadowpa0327/Palu`（arXiv 2407.21118）的秩搜索，`search_method == "fisher"` 分支：

- `palu/rank_search.py::rank_search` L105-168：L120 `module_rank = info.num_lr_groups * info.lr_group_dims`、L123 初值 `[info.lr_group_dims] * num_lr_groups`（这就是上限）；L125 `fisher.reshape(num_lr_groups, -1, in_features)`、L129 `torch.mean(fisher[i])`（组 Fisher = 整块均值）、L131 `fisher_sum`；L134 `target_rank = total_rank * param_ratio_target`、L142 `rank_float = target_rank * fisher[i] / fisher_sum`、L146 `min(select_result[name][i], math.floor(rank_float))`；L150 `indexes = sorted(indexes, key=rank_float - floor)`、L151 `dif`、L154-163 `while dif > 0` 的轮转（满组 `continue`）；L164 取整、L165-166 `rank_sum / total_rank`。
- `palu/rank_search.py::rounding_search_result` L11-17：L11 `block_size=32`，L15 `max(1, round(ranks[i] / block_size)) * block_size`——Python 内建 `round` 本来就是 half-to-even。
- 分组语义 `palu/model/svd_llama/modeling_palu_llama.py::get_kv_info` L38-59：L40 `num_lr_groups = num_key_value_heads // head_group_size`、L42 `lr_group_dims = head_dim * head_group_size`、L44-54 不整除即 `ValueError`。
- 秩的消费端（本题不碰）：`palu/decomposition.py::compress_model_svd` L228-259 → `palu/model/modules/svd_linear.py::HeadwiseLowRankModule.from_linear` L206-235（L212 按组 reshape 权重、L218 每组 SVD 截到 `ranks[i]`）、L72 `nn.Linear(in_features, sum(ranks))` 即 latent 宽 = Σ秩、`fused_hadamard_matrix` L156-168 是配 latent 量化的 Hadamard 旋转。

**本题自己的补充**（上游没写或写得不同，题面显式钉死）：一次只结算一个模块、一段预算（上游 L131 把 k/v  proj 与所有层的 Fisher 累成一个全局 `fisher_sum`），并把它压成单个 `(G*head_group_size, lr_group_dims)` 矩阵，列宽语义由上游的 `in_features` 改成 `lr_group_dims`；`head_group_size`/`lr_group_dims`/`block_size` 全部走显式入参；补发次数钉成 `floor(dif)`——上游那个 for 只在 `dif == 0` 时才 break，`dif` 是小数时会把整趟走完而多发；加 `need > headroom` 守卫（上游全是满组时会死循环）；同余数按组号升序（稳定排序）；形状/dtype/ratio/非正的整套校验；输出六个中间量而不是 `logger.info` 一行压缩率。

## 相关题目

- `kv-xkv-layer-svd-share`：同样是低秩共享，但颗粒是**跨层的奇异向量**；本题的共享只发生在组内的 head 之间，且只算预算不算基。
- `kv-layer-budget-knapsack`：层间保留名额用背包求最优；本题是 Fisher 比例 + 整数补齐的贪心账，不比较方案。
- `kv-quest-page-bound`：decode 期在线选页的预算；本题是离线一次性分秩，分完就固化进权重。
