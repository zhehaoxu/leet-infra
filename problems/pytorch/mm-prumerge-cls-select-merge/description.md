# 多模态：LLaVA-PruMerge 的 CLS 注意力选 token 与余弦残差合并

## 任务

在 CLIP 视觉塔内部、投影层之前，用 CLS 对每个 patch 的注意力选出少数「关键 patch」，把其余 patch 按**余弦相似度**分配给最像自己的关键 patch，再把它们的特征**加权残差合并**进对应的关键 patch，从而只保留约 `1/8` 的视觉 token。

本题有意抽取 LLaVA-PruMerge（Shang et al., arXiv:2403.15388）官方实现的选择与合并两步，**不是完整方法**：patch 特征与 CLS→patch 注意力都由调用方给出，模型不在此处前向。参数、形状、数值范围、返回顺序与异常约定统一见 starter 的双语契约，以下只定义计算规则。

## 选择规则

记 `N` 为 patch 数，`C` 为特征维，`k` 为保留个数。

1. `k = int(N * reduction_ratio)`：使用**向下截断**，不是四舍五入，因此 `int(5 * 0.5) = 2`、`int(3 * 0.5) = 1`。若 `k` 截断成 `0` 则抛 `ValueError`。
2. 保留 `cls_attn` 最大的 `k` 个 patch。并列按「值降序、下标升序」判定；最终保留下标必须**按升序**发出，被合并掉的下标同样按升序发出，两组互不重叠且并起来恰为 `0..N-1`。

## 合并规则

对每一行独立执行，`i` 枚举升序的 pruned patch，`j` 枚举升序的 kept patch（即 `kept_indexs` 的列号）：

1. 归一化分母取 `max(||x||_2, eps)`，`eps` 默认 `1e-12`，与 `F.normalize` 的下限语义一致。零向量的单位向量因此是全零，其余弦为 `0` 而不是 `NaN`。
2. 相似度 `sim[i,j] = <u_i, u_j>`，其中 `u` 是上一步的单位向量。
3. 分配 `cluster[i] = argmax_j sim[i,j]`，并列取**第一个**（最小的 `j`）。
4. 权重 `p[i,:]` 是 `sim[i,:]` 沿**整个 kept 集合**的稳定 softmax，然后
   $$
   \mathrm{weight}[i] = p[i,\mathrm{cluster}[i]] \cdot C^{-1/2}.
   $$
   缩放发生在 softmax **之后**，用的是特征维 `C`，不是 token 数；同一 cluster 内的权重**不再归一化**，所以一个 cluster 的权重和通常不等于 `1`。
5. 合并是残差形式：
   $$
   \mathrm{merged}[j] = x_{\mathrm{kept}[j]} + \sum_{i:\,\mathrm{cluster}[i]=j} \mathrm{weight}[i]\cdot x_{\mathrm{pruned}[i]}.
   $$
   没有被分配任何 pruned patch 的 kept patch **原样返回**。`reduction_ratio = 1` 时 pruned 为空，输出就是升序的原始 patch。

`scores[i] = sim[i, cluster[i]]` 即每个 pruned patch 的最大余弦，作为可判定的台账一并返回。

## 样例 1：三合一 cluster 与空 cluster

全部浮点张量为 float64，`N = 5`，`C = 2`，`reduction_ratio = 0.5`，`eps = 1e-12`，故 `k = int(5 * 0.5) = 2`。

```text
image_features = [[[1, 1], [1, 0], [2, 0], [0, 1], [1, -1]]]
cls_attn = [[0.2, 0.9, 0.4, 0.7, 0.1]]

merged        = [[[2.956244, -0.215264], [0.000000, 1.000000]]]
kept_indexs   = [[1, 3]]
pruned_indexs = [[0, 2, 4]]
cluster       = [[0, 0, 0]]
weights       = [[0.353553, 0.516936, 0.568818]]
scores        = [[0.707107, 1.000000, 0.707107]]
```

注意力最高的两个 patch 是下标 `1` 和 `3`，升序成为两个 kept 列。三个 pruned patch 全部落到列 `0`：下标 `0` 与列 `0`、列 `1` 的余弦并列 `0.707107`，取第一个；它的权重是 softmax 在该列的取值 `0.5` 再乘 `C ** -0.5 = 0.707107`。列 `1` 没有分配到任何 patch，因此逐字节等于原始下标 `3` 的特征 `[0, 1]`。三个权重之和约 `1.439307`，而不是 `1`，这正是「softmax 覆盖全部 kept、cluster 内不再归一化」的证据。

## 样例 2：截断、零向量与 `C ** -0.5`

全部浮点张量为 float64，`N = 4`，`C = 4`，`reduction_ratio = 0.25`，`eps = 1e-12`，故 `k = int(4 * 0.25) = 1`。

```text
image_features = [[[1, 1, 0, 0], [0, 0, 0, 0], [2, 2, 0, 0], [1, 0, 1, 0]]]
cls_attn = [[0.6, 0.6, 0.9, 0.2]]

merged        = [[[3.000000, 2.500000, 0.500000, 0.000000]]]
kept_indexs   = [[2]]
pruned_indexs = [[0, 1, 3]]
cluster       = [[0, 0, 0]]
weights       = [[0.500000, 0.500000, 0.500000]]
scores        = [[1.000000, 0.000000, 0.500000]]
```

只有一个 kept 列，所以 softmax 退化为 `1`，每个权重恰为 `C ** -0.5 = 0.5`。下标 `1` 是零向量：分母被 `eps` 兜底，余弦为 `0`、不产生 `NaN`，它的贡献是 `0.5` 乘零向量。合并结果 `[2, 2, 0, 0] + 0.5 * ([1, 1, 0, 0] + [0, 0, 0, 0] + [1, 0, 1, 0]) = [3, 2.5, 0.5, 0]`。

以上输出均由参考实现实跑，并通过独立 oracle 校验；展示值保留六位小数。
