# ToMe 的二分软匹配合并账：余弦精确比较、token 归并与尺寸加权归一

## 背景

Token Merging（ToMe）不删 token，而是**把相似的 token 合并成一个**：对每个 token 找它的「合并对象」，让被合并的那个吸收对方。这样 token 数少了，但每个 token 代表的信息量变多。

官方 `facebookresearch/ToMe` 的做法是**二分软匹配**（bipartite soft matching）：把 token 序列按奇偶位置**交错分半**成两组 `a`（偶数位）与 `b`（奇数位），只在 `a` 的 token 里挑出要消失的 `r` 个（`src`），每个 `src` 找一个最相似的 `b`（`dst`），合并时 `dst` 吸收 `src`。剩下的 `unm`（未合并的 `a`）与 `b` 拼起来就是下一层的 token 序列。

这条路径里有几处极容易写错的细节：

- **`-inf` 保护在两个不同的轴上**：`class_token` 保护的是**第 0 行**（`a` 侧的第 0 个），`distill_token` 保护的是**第 0 列**（`b` 侧的第 0 个）。写成同一轴就全错。
- **`scatter_reduce` 的 `mean` 分母含目标自身**（`include_self=True`），不是「落进来的 src 个数」。
- **`merge_wavg` 的分母是 size 之和**，不是 token 个数。
- **官方是浮点实现**，在「精确余弦完全相等」处，1e-17 的尾差会让 `max` 与 `argsort` 翻转选中项——同一份输入在不同机器上可能得到不同的合并方案。

本题重建这条路径的**纯数值**账。为了让账本可判定，题面把输入限定为整数，并把相似度比较钉成**精确有理数**比较（不开根号，用整数交叉相乘），并列规则也一并钉死。ViT 前向、GPU kernel、timm patch 的实际接线都不在判题域内。

## 通用约定

- 所有输入都是 Python `int` / `float` 组成的嵌套列表。不涉及张量、梯度、GPU。
- `metric` 的形状是 `[B][T][C]`（batch × token × 通道），元素是**整数**，且每一行的模长**非零**（否则余弦无定义）。
- `x` 的形状是 `[B][T][C]`，元素是整数；`size` 的形状是 `[B][T][1]`，元素是正整数。
- 下标一律从 0 开始，且**闭区间**。
- `T` 记作 token 数，`C` 记作通道数，`B` 记作 batch 数。
- 返回值统一是 `dict`，字段名见各 mode 说明。
- 所有 mode 都通过单入口 `solve(mode, **kw)` 调用。

## 精确余弦比较（本题的核心口径）

官方算的是 `metric / metric.norm(dim=-1)` 之后做 `a @ b.T`，得到余弦相似度。本题把它换成**精确有理数**口径：

```
s_ij   = Σ_c a_i[c] * b_j[c]        # 精确整数点积
m2_i   = Σ_c a_i[c]^2                # 精确整数模长平方
cos_ij = s_ij / sqrt(m2_i * m2_j)   # 真实值
```

比较两个候选 `u = cos_ij` 与 `v = cos_pq` 时**不开根号**，用整数交叉相乘：

```
先判符号：非负 > 负
都非负：u >= v  <=>  s_ij^2 * m2_p * m2_q >= s_pq^2 * m2_i * m2_j
都为负：u >= v  <=>  s_ij^2 * m2_p * m2_q <= s_pq^2 * m2_i * m2_j
```

**并列规则**（官方浮点在此处未定义，本题钉死）：

- `max` 取到并列最大值时选**下标最小**者；
- 按相似度**降序**排序视为**稳定**排序，并列时保持原下标升序。

被 `-inf` 保护的位置（见下）视为负无穷，排在最后，且永远不会被选为 `dst`。

## 模式与规则

### 1. `match_indices`：二分软匹配的索引计算

入参：

| 参数 | 含义 | 默认值 |
|---|---|---|
| `metric` | 三维列表 `[B][T][C]`，元素是整数 | 必填 |
| `r` | 要合并掉的 token 数 | 必填 |
| `class_token` | 是否有 class token | `False` |
| `distill_token` | 是否有 distillation token | `False` |

计算步骤：

```
# 1. 有效 r：最多合并一半（还要扣掉被保护的 token）
protected = int(class_token) + int(distill_token)
r_effective = min(r, (T - protected) // 2)

# 2. 交错分半：a 侧是偶数位，b 侧是奇数位
t1 = (T + 1) // 2      # a 侧长度
t2 = T // 2            # b 侧长度
a = metric[:, 0::2, :] # 偶数位
b = metric[:, 1::2, :] # 奇数位

# 3. 相似度矩阵（精确口径），零模长的行/列视为负无穷
scores[i][j] = cos(a_i, b_j)

# 4. -inf 保护在两个不同的轴上
if class_token:
    scores[0][:] = -inf        # 第 0 行（a 侧第 0 个）整行
if distill_token:
    scores[:][0] = -inf        # 第 0 列（b 侧第 0 个）整列

# 5. 每个 a 元素选最相似的 b（并列取下标最小）
node_max[i], node_idx[i] = max_j scores[i][j]

# 6. 按 node_max 降序稳定排序，取前 r_effective 个作为 src
order = stable_argsort_desc(node_max)
src_idx = order[:r_effective]
unm_idx = order[r_effective:]
dst_idx = [node_idx[i] for i in src_idx]

# 7. 只有 class_token 时才把 unm_idx 排序（保证 class token 在最前）
if class_token:
    unm_idx = sorted(unm_idx)
```

**`r_effective <= 0` 的特例**：官方此时直接返回 `do_nothing`（不做任何匹配、也不合并）。本题口径：

- `src_idx` / `dst_idx` 是空列表；
- `unm_idx` 是 `[0, 1, ..., t1-1]`（a 侧全部下标，升序）；
- `node_max` 是空列表（不计算）。

返回字段：

- `r_effective`：实际生效的 `r`；
- `src_idx`：长度 `B`，第 `b` 项是长度 `r_effective` 的**a 侧下标**列表（按相似度降序）；
- `dst_idx`：长度 `B`，第 `b` 项是长度 `r_effective` 的**b 侧下标**列表（每个 `src` 的匹配目标）；
- `unm_idx`：长度 `B`，第 `b` 项是未合并的 a 侧下标列表；
- `node_max`：长度 `B`，第 `b` 项是长度 `t1` 的列表，第 `i` 项是 `a_i` 与其最佳匹配的余弦值（被保护或零模长时为 `-inf`）；
- `num_src`：`t1`，即 a 侧长度；
- `num_dst`：`t2`，即 b 侧长度。

易错点：

- **保护轴不能搞混**。`class_token` → `scores[0][:] = -inf`（**行**）；`distill_token` → `scores[:][0] = -inf`（**列**）。`scores[0][:]` 被清空时，`a_0` 只能当 `src` 不能当 `dst`……更准确地说：行保护让 `a_0` 不会被选为任何 `b` 的匹配源（它的 `node_max` 是 `-inf`，排序垫底）；列保护让 `b_0` 不会被任何 `a` 选中。
- **`class_token` 与 `distill_token` 都开时，`protected = 2`**，所以 `r_effective` 的上限是 `(T - 2) // 2`，比只开一个更严。
- **只有 `class_token` 会触发 `unm_idx` 的排序**。`distill_token` 单独开时 `unm_idx` 保持相似度降序，不排序。这个「只有 class_token 才排序」的条件很容易写反。
- **`dst_idx` 允许重复**：多个 `src` 可能匹配到同一个 `b`（它们最相似的 `b` 相同）。这是官方语义，不是 bug。
- **`node_max` 的值是余弦**，不是点积也不是归一化后的向量；被保护的行是 `-inf`。
- **`r` 为负数时 `r_effective` 也是负数**（`min` 不做钳位到 0），判题按原值返回。
- **奇数 `T` 时 `a` 侧比 `b` 侧多一个**（`t1 = (T+1)//2`，`t2 = T//2`），分半是按**原序列的偶数/奇数位**而不是「前一半/后一半」。

### 2. `merge_reduce`：合并（gather + scatter_reduce + 输出重排）

入参：

| 参数 | 含义 | 默认值 |
|---|---|---|
| `x` | 三维列表 `[B][T][C]` | 必填 |
| `src_idx` | 被合并的 a 侧下标（长度 `B`） | 必填 |
| `dst_idx` | 每个 `src` 的 b 侧目标（长度 `B`） | 必填 |
| `unm_idx` | 未合并的 a 侧下标（长度 `B`） | 必填 |
| `reduce` | `"mean"` / `"sum"` / `"amax"` | `"mean"` |
| `distill_token` | 是否用交错输出布局 | `False` |

计算步骤：

```
# 1. 奇偶分半
src_rows = x[:, 0::2, :]    # a 侧
dst_rows = x[:, 1::2, :]    # b 侧

# 2. gather
unm = src_rows.gather(unm_idx)     # 未合并的 a 侧 token
src = src_rows.gather(src_idx)     # 被合并的 a 侧 token

# 3. scatter_reduce 到 b 侧（include_self=True，官方默认）
dst = dst_rows.scatter_reduce(-2, dst_idx, src, reduce=reduce)

# 4. 输出重排
if distill_token:
    merged = cat([unm[:1], dst[:1], unm[1:], dst[1:]])
else:
    merged = cat([unm, dst])
```

`scatter_reduce` 的三种归约语义（`include_self=True`，即**目标自身的旧值参与归约**）：

| `reduce` | 目标位置 `j` 的结果 | 语义 |
|---|---|---|
| `"sum"` | `dst_old[j] + Σ(src[i] for i 落在 j)` | 求和 |
| `"mean"` | `(dst_old[j] + Σ src[i]) / (1 + 落在 j 的 src 个数)` | 均值，**分母含目标自身** |
| `"amax"` | `max(dst_old[j], max(src[i] for i 落在 j))` | 最大值 |

返回字段：

- `merged`：三维列表 `[B][T - r][C]`，合并后的 token 序列（`r` 是 `src_idx[b]` 的长度）；
- `merged_len`：合并后的 token 数；
- `dst_values`：三维列表 `[B][t2][C]`，scatter_reduce 之后的 b 侧值；
- `num_src`：`t1`；
- `num_dst`：`t2`。

易错点：

- **`mean` 的分母是「1 + 落进来的 src 个数」**。若两个 `src` 落到同一目标且目标旧值是 `10`、两个 `src` 是 `1` 和 `2`，`mean` 得 `(10 + 1 + 2) / 3 = 4.333`，**不是** `1.5`。
- **`include_self=False` 时才是 `(1 + 2) 分母不含目标`**。官方用的是默认的 `True`。
- **`src_idx` 为空时官方返回 `do_nothing`**（恒等）：`merged` 就是输入 `x` 本身，`merged_len = T`，`dst_values` 是未合并的 b 侧原值。
- **`distill_token` 的交错布局**：输出是 `[unm[0], dst[0], unm[1], dst[1], ...]`，不是 `[unm..., dst...]`。这是为了让 distillation token 与 class token 在序列里相邻。
- **`merged` 的长度是 `T - r`**，不是 `T - 2r`（只合并掉 `r` 个 a 侧 token，b 侧数量不变）。

### 3. `size_weighted_merge`：尺寸加权合并（`merge_wavg`）

入参：

| 参数 | 含义 | 默认值 |
|---|---|---|
| `x` | 三维列表 `[B][T][C]` | 必填 |
| `size` | 三维列表 `[B][T][1]`，每个 token 代表的原始 token 数 | 必填 |
| `src_idx` / `dst_idx` / `unm_idx` | 同 mode 2 | 必填 |
| `distill_token` | 是否用交错输出布局 | `False` |

计算步骤：

```
# 1. 先按 size 加权，再做「sum」归约
scaled = x * size                     # 逐元素乘
x_merged = merge(scaled, reduce="sum")

# 2. size 本身也用同一套合并规则归约
size_merged = merge(size, reduce="sum")

# 3. 除以归约后的 size
merged = x_merged / size_merged
```

返回字段：

- `merged`：三维列表 `[B][T - r][C]`，加权平均后的 token；
- `merged_size`：三维列表 `[B][T - r][1]`，归约后的 size；
- `merged_len`：合并后的 token 数。

易错点：

- **分母是 size 之和，不是 token 个数**。一个 size 为 `3` 的 token 单独留在序列里时，它的分母是 `3`（不是 `1`）。
- **`x` 与 `size` 走的是同一套合并规则**（包括 `distill_token` 的交错布局），所以 `merged` 与 `merged_size` 的**第 `i` 行严格对应**——除法是逐行对齐的。
- **`size` 的形状是 `[B][T][1]`**（通道维长度为 1），不是 `[B][T]`。
- **`src_idx` 为空时是 `do_nothing`**：`merged` 等于 `x`、`merged_size` 等于 `size`（**不是** `x * size / size` 的逐项结果——虽然数值上 `x*size/size` 也等于 `x`，但 `merged_size` 必须是原 `size` 而不是 `size * size / size` 之后的形态；官方 `merge` 是恒等函数，`size` 不变）。

### 4. `unmerge`：位置回填（把合并后的序列还原成原布局）

入参：

| 参数 | 含义 | 默认值 |
|---|---|---|
| `x` | **合并后**的三维列表 `[B][T - r][C]` | 必填 |
| `src_idx` / `dst_idx` / `unm_idx` | 同 mode 2 | 必填 |

计算步骤：

```
# 1. 按 unm 长度切分：前半是未合并的 a 侧 token，后半是 b 侧 token
unm_len = len(unm_idx)
unm = x[:, :unm_len, :]
dst = x[:, unm_len:, :]

# 2. 每个 src 的值 = 它合并到的 dst 的值（合并后的值，不是原值）
src = dst.gather(dst_idx)

# 3. 回填到原始布局
out = zeros(B, T, C)
out[:, 1::2, :] = dst            # b 侧 token 回到奇数位
out[2 * unm_idx] = unm           # 未合并的 a 侧 token 回到偶数位
out[2 * src_idx] = src           # 被合并的 a 侧 token 也回到偶数位（值是合并后的）
```

返回字段：

- `out`：三维列表 `[B][T][C]`，还原后的序列；
- `out_len`：原始 token 数 `T`（即 `x` 的 token 数加上 `r`）。

易错点：

- **输入是合并后的张量**（长度 `T - r`），输出才是 `T` 行。不要把合并前的序列传进来。
- **`out` 里被合并的 `src` 位置填的是合并后的值**（`dst` 的值），不是它自己原来的值——`unmerge` 恢复的是**布局**，不是数值。
- **`2 * unm_idx` / `2 * src_idx` 是原序列的偶数位下标**（因为 `a` 侧来自偶数位）。
- **被合并的 `src` 与未合并的 `unm` 在偶数位上互补**，所以两轮 `scatter` 不会互相覆盖。

## 判题范围

- 只判上面四段纯数值账：`match_indices` / `merge_reduce` / `size_weighted_merge` / `unmerge`。
- **不判**：ViT 前向、`timm` patch 的实际接线（`ToMeBlock` 在 attention 与 mlp 之间插入合并）、比例注意力（`attn + size.log()`）、`parse_r` 的多层 `r` 调度、`merge_source` 的来源矩阵、`kth_bipartite_soft_matching` 与 `random_bipartite_soft_matching` 两个变体、GPU 上的实际加速比与精度损失、真实模型的权重。
- 题面里的 `metric` / `x` / `size` 只是数值列表，**不含任何真实特征**。
