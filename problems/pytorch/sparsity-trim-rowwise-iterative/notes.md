# 解题说明 · TRIM 逐行稀疏率迭代

## 一句话

用「剪枝前后输出行的余弦」当反馈，每轮把稀疏预算从输出变化小的行挪给变化大的行，外层再用学习率网格挑一轮最好的结果。

## 动机与目标

逐行统一稀疏率隐含假设「每一行一样耐剪」。TRIM 的出发点是：不同输出行对同一比例的删除反应不同，行的稀疏率应当由**该行剪枝后的输出保真度**决定。因此它不引入新的重要性度量，而是直接在「输出空间」里做闭环：哪一行的余弦掉了，就减少该行的稀疏率（少剪），把预算转给余弦仍然接近 1 的行。

## 算法要点

1. **排序固定**。`sorted_idx` 由 `|W|·sqrt(act_ms)` 稳定升序得到，剪枝永远是前缀，所以每轮只改变「剪多深」。
2. **归一化增量**。`d=(q-min)/(max-min+1e-6)` 把逐行余弦映射到 `[0,1]`，乘 `2η`、减均值后加到 \(x\)。减均值保证这一步本身不改变平均预算；随后的 `x-mean(x)+target` 再次把均值钉回目标。
3. **硬截断**。`x` 先夹到 `[0, 0.95]` 再重新对齐；`0.95` 是上游防止「整行剪空」的稳定上限。
4. **择优指标**。整层扁平余弦（`cosim_flat`）衡量「剪枝后整层输出方向」的保真度；每轮都重新剪、重新乘 `inputs`，只在严格提升时记录 `best_x` 与 `best_iter`。
5. **学习率扫描**。正网格逐个试，一旦某次不再提升就停（网格是递增的，继续试更大的步长没有意义）；正网格全军覆没时才试负网格（反向调整稀疏率）。`learning_rate` 可以为 `None`。
6. **后处理**。最优 \(x\) 再夹一次、重新对齐后生成掩码；返回的 `sparsities` 是后处理后的值，而 `best_metric`/`learning_rate` 记录的是后处理前那次择优。

## 判分点

- `keep_mask` 每行被剪数量恰为 `int(in·s_i)`（`s_i` 为最终稀疏率，负值按 0、超过 1 按全部处理）；
- `sparsities.mean() == target_sparsity`（截断会破坏时由第二次对齐拉回）；
- `best_metric >= baseline_metric`，且 `learning_rate` 非空时严格大于；
- `starting_var`/`ending_var` 用无偏方差；`best_iter` 是 0 起的轮次；
- 输入不被就地修改；`pruned_weight` 只改被剪位置。

## 与相邻题目的边界

- `sparsity-wanda-activation-metric`：一次性、行稀疏率全部相同的 Wanda；本题在其排序之上**迭代出不同的行稀疏率**。
- `sparsity-owl-outlier-layerwise`：在**层间**分配预算，层内仍统一；本题在**行间**分配预算。
- `sparsity-movere-reactivation`：动态稀疏训练里按梯度改掩码；本题是训练后一次性求解，不碰梯度。

## 出处

`flobk/TRIM`（arXiv 2505.16743）：

- `lib/trim.py::calculate_row_wise_sparsities`：正/负学习率网格、`best_metric`/`best_learning_rate`/`best_iter` 的更新与提前停止；
- `lib/trim.py::get_sparsities_iterative`：`unpruned_outputs = W @ inputs`、`update_x` 的 `(qm-min)/(max-min+1e-6)`、`*learning_rate*2`、`-mean`、`x>0.95→0.95`、`x<0→0`、`x-mean+target`；
- `lib/trim.py::prune_matrix`：`num_prune = (width*sparsities).long()`、`mask[row, sorted_idx] = ~prune_mask`；
- `lib/trim.py::cosine_similarity*`：`1e-8` 范数保护、`clamp(-1,1)`；
- `lib/prune.py` 的 TRIM 分支：`W_metric = |W|*sqrt(scaler_row)`、`sorted_idx = sort(..., stable=True)[1]`、调用后 `>0.95→0.95`、`<0→0`、`-mean+ratio`。

**本题自己的补充**：把 `inputs` 与 `activation_ms` 都做成显式入参（上游前者取末 token、后者取全 token 统计）；排序并列取小下标写成硬约定；只保留 `cosim_flat` 一种择优指标；`out>=2` 与 `target∈(0,1)` 作为显式校验。
