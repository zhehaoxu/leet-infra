# 解题说明 · OWL 离群比驱动的逐层稀疏率分配

## 一句话

先把每层所有 Linear 的 `|W|·RMS` 拼成一个大向量，数一数有多少元素超过「均值 × hyper_m」，得到离群比；再把离群比 min-max 归一到 `[0, 2λ]`、减去均值、对齐到 `1-s`，得到逐层密度；最后层内按行剪最小的 `int(in·s_l)` 个。

## 为什么用离群比分配预算

全局稀疏率固定时，预算在层间怎么分是自由变量。OWL 的观察是：一层的激活越「重尾」，其 metric 分布越被少数大值拉高，阈值 `mean·hyper_m` 之上的元素比例就越大；这类层对总误差更敏感，应该少剪。于是用离群比当层分数，min-max 归一后按分数分配密度。减去均值这一步保证**平均密度回到 `1-s`**，即预算总量不变。

## 三个容易写错的地方

1. **按层拼接，不按 Linear 拼接**。`layer_wmetric.append(W_metric)` 之后 `torch.cat([flatten(x) ...])`，均值和阈值都取自整层；层内每个 Linear 只是共用得到的层稀疏率。
2. **严格大于**。`(W > max_shred).sum()`，等号不算。`[1,1,1,9]` 的均值是 3、阈值是 9，9 不是离群。
3. **下限 0.01 的含义**。`if layer_sparsity_ratio<=0: layer_sparsity_ratio=0.01`，是把稀疏率设成 1%，不是把密度夹到 1%。稀疏率超过 1 的情况没有做上限截断：`int(in·s)` 大于 `in` 时切片自然返回全部列，等价于整层剪空。

## 判分点

- `outlier_ratio`、`density`、`layer_sparsity` 是 float64，形状 `(L,)`；
- `metric`/`pruned_weight` 与输入同 dtype、同嵌套结构，`mask` 是 bool 且 True=剪；
- 同一层多个 Linear 的层稀疏率相同，`int(width·s_l)` 各自计算；
- `layer_sparsity.mean() == sparsity`（没有触发 0.01 下限时）；
- 各层离群比相同时抛 `ValueError`，不返回 NaN。

## 与相邻题目的边界

- `sparsity-wanda-activation-metric`：只做单层 metric + 逐行掩码；本题在它之上加**跨层的预算分配**。
- `kv-pyramid-layer-budget`（kvcache）：按层深度线性分配 KV 预算；OWL 按离群统计分配**权重剪枝**预算，统计量与对象都不同。
- `sparsity-rowwise-prune`：所有层同一稀疏率的逐行剪枝。

## 出处

`luuyin/OWL`（arXiv 2310.05175）：

- `lib/prune_all.py::prune_wanda_outlier`：`W_metric = |W| * sqrt(scaler_row.reshape((1,-1)))`；`layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])`；`out_ratio_layer = check_outlier_mean(layer_wmetric, args.Hyper_m)`；
- `check_outlier_mean`：`max_shred = torch.mean(W)*threshold`、`count += (W > max_shred).sum()`、`outlier_ratio = count/total_params*100`；
- 预算：`(r - r.min()) * (1/(r.max()-r.min()) * args.Lamda*2)`、`- np.mean(...) + (1-args.sparsity_ratio)`、`layer_sparsity_ratio = 1 - all_layer_ratio[i]`、`<=0 → 0.01`；
- 掩码：`sort(W_metric, dim=-1, stable=True)` 后取前 `int(cols*layer_sparsity_ratio)`。

默认值来自 `main.py`：`Lamda=0.08`、`Hyper_m=3`、`sparsity_ratio` 必须显式给。

**本题自己的补充**：结构显式化为 `[层][Linear]` 嵌套列表；同分取小下标写成硬约定；各层离群比相同时显式报错而不是产生 NaN；返回逐层 `density` 便于对拍。
