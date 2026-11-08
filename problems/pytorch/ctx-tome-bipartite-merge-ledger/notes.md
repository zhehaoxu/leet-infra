# 官方机制与易错点笔记

## 官方来源与版本

- 仓库：`facebookresearch/ToMe`，默认分支 `main`（GitHub API trees 返回 1205 stars 量级的活跃仓库）。
- 用到的文件：
  - `tome/merge.py`（240 行）：`do_nothing` / `bipartite_soft_matching` / `kth_bipartite_soft_matching` / `random_bipartite_soft_matching` / `merge_wavg` / `merge_source`；
  - `tome/utils.py`：`parse_r`；
  - `tome/patch/timm.py`（150 行）：`ToMeBlock.forward` / `ToMeAttention.forward` / `apply_patch`。
- 回源方式：GitHub API `git/trees/main?recursive=1` + `contents?ref=main` + `base64.b64decode`。
- 官方对拍脚本用 `.venv-eval/bin/python`（torch 2.14.1，无需 GPU，官方实现是纯张量运算）。

## 官方 `bipartite_soft_matching` 逐行解剖

```python
protected = 0
if class_token:  protected += 1
if distill_token: protected += 1
t = metric.shape[1]
r = min(r, (t - protected) // 2)
if r <= 0:
    return do_nothing, do_nothing
with torch.no_grad():
    metric = metric / metric.norm(dim=-1, keepdim=True)
    a, b = metric[..., ::2, :], metric[..., 1::2, :]
    scores = a @ b.transpose(-1, -2)
    if class_token:  scores[..., 0, :] = -math.inf
    if distill_token: scores[..., :, 0] = -math.inf
    node_max, node_idx = scores.max(dim=-1)
    edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
    unm_idx = edge_idx[..., r:, :]
    src_idx = edge_idx[..., :r, :]
    dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
    if class_token:
        unm_idx = unm_idx.sort(dim=1)[0]
```

逐条钉死的机制：

1. **`-inf` 保护在两个不同的轴上**：`class_token` → `scores[..., 0, :]`（**行**）；`distill_token` → `scores[..., :, 0]`（**列**）。这是本题最主要的判题点，写成同一轴就全错。
2. **`r` 的上限是 `(t - protected) // 2`**：`protected` 是 `class_token` 与 `distill_token` **各占一个**。两者都开时上限是 `(t - 2) // 2`。
3. **`r <= 0` 时返回 `do_nothing`**（`lambda x, mode=None: x`），即**完全恒等**：不匹配、不合并、长度不变。
4. **交错分半**：`a = metric[..., ::2, :]`（偶数位）、`b = metric[..., 1::2, :]`（奇数位）。`t` 为奇数时 `a` 比 `b` 多一个。
5. **`unm_idx` 只在 `class_token` 时排序**（`unm_idx.sort(dim=1)[0]`，沿 token 维升序），目的是让 class token 排在最前。`distill_token` 单独开时**不排序**。
6. **`src_idx` 与 `unm_idx` 是 `edge_idx` 的互补两段**：`src_idx` 是前 `r` 个（按相似度降序），`unm_idx` 是剩下的。
7. **`dst_idx` 允许重复**：多个 `src` 可能匹配同一个 `b`。

## 官方 `merge` 闭包解剖

```python
def merge(x, mode="mean"):
    src, dst = x[..., ::2, :], x[..., 1::2, :]
    n, t1, c = src.shape
    unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
    src = src.gather(dim=-2, index=src_t.expand(n, r, c))
    dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
    if distill_token:
        return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
    return torch.cat([unm, dst], dim=1)
```

- **`scatter_reduce` 的 `include_self` 默认是 `True`**。实测（`dst=[10,20,30]`、`src=[1,2,4]`、`idx=[0,0,1]`）：
  - `sum` → `[13, 24, 30]`；
  - `mean` → `[4.333, 12, 30]`（即 `(10+1+2)/3`，**分母含目标自身**）；
  - `amax` → `[10, 20, 30]`（含目标自身）。
  - `include_self=False` 时 `mean` 才是 `[1.5, 4.0, 30.0]`。
  - 实测重复索引在 CPU 上是**确定性**的（多次运行结果一致）。
- **`distill_token` 的交错布局**：`cat([unm[:1], dst[:1], unm[1:], dst[1:]])`，让 distillation token 与 class token 相邻。
- **输出长度是 `t - r`**（只合并掉 `r` 个 a 侧 token）。

## 官方 `merge_wavg` 解剖

```python
def merge_wavg(merge, x, size=None):
    if size is None:
        size = torch.ones_like(x[..., 0, None])
    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")
    x = x / size
    return x, size
```

- **分母是 size 之和，不是 token 个数**。一个 size 为 `3` 的 token 单独留下时，它的分母是 `3`。
- **`x` 与 `size` 走同一个 `merge` 闭包**，所以 `distill_token` 的交错布局对两者都生效——`merged` 与 `merged_size` 逐行对应。
- **`size` 的形状是 `[..., 0, None]`**，即 `[B, T, 1]`。

## 官方 `unmerge` 闭包解剖

```python
def unmerge(x):
    unm_len = unm_idx.shape[1]
    unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
    src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))
    out = torch.zeros(n, metric.shape[1], c, ...)
    out[..., 1::2, :] = dst
    out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
    out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)
    return out
```

- **输入是合并后的张量**（长度 `t - r`），输出是 `t` 行。`out` 的长度取自 `metric.shape[1]`（原 token 数）。
- **`src` 的值是合并后的 `dst` 值**（不是它自己的原值）——`unmerge` 恢复的是**布局**不是数值。
- **`2 * unm_idx` / `2 * src_idx`** 是原序列的偶数位下标。

## 浮点尾差与并列：本题口径的由来

官方是 float64 的 `metric / metric.norm()` + `a @ b.T`。实测发现三件事：

1. **torch `max(dim=-1)` 在并列时取最小下标**（`[1,1,1,-1]` → index 0）；
2. **torch `argsort(dim=-1, descending=True)` 在 CPU 上保序稳定**（`[0.5,0.5,0.5,0.9,0.5]` → `[3,0,1,2,4]`）；
3. **归一化会引入尾差**：`[1,1]` 与 `[2,2]` 归一化后点积是 `0.9999999999999998` 而不是 `1.0`。

第 3 点是致命伤：在「精确余弦完全相等」的 case 上，官方浮点会因为 1e-17 尾差打破并列，于是 `max` 选中的项与 `argsort` 的顺序**依赖于尾数**——同一份输入在不同 BLAS / 不同机器上可能得到不同的合并方案。

因此本题把口径钉成**精确有理数**：`metric` / `x` / `size` 全整数，比较时用整数交叉相乘消根号，并列规则显式写死（`max` 取下标最小、排序稳定保序）。

**这个决定带来的连带要求**：官方浮点与精确口径在**非退化** case 上完全一致，只在**精确并列**处分歧。官方交叉验证必须按这个事实设计——不能拿官方浮点当逐点 oracle，而要**定位分歧点并证明它的精确余弦差 <= 1e-12**（数学上确实相等，浮点噪声导致翻转）。详见下节。

## 验证方式与判题设计

- **官方回源对拍（三层）**：
  1. **主 oracle**：纯 Python + `fractions.Fraction` 的精确有理数实现，与参考解的「整数交叉相乘」是**两条独立代码路径**（Fraction 直接做精确除法比较）；
  2. **官方交叉验证**：官方 `tome/merge.py` 的 float64 实现。`match_indices` 600 组中 **567 组索引完全一致**，另 **33 组分歧**全部被定位为「精确余弦几乎相等（<= 1e-12，浮点尾差）」，无一例外——即官方与精确口径的差异**只发生在数学上确实并列的地方**；
  3. **merge 机制交叉验证**：直接用官方的 `gather` / `scatter_reduce` / `cat` / `scatter_` 语义跑我们精确算出的索引，验证 gather/scatter_reduce 的 `include_self` 语义、输出重排、`distill_token` 的交错布局、`merge_wavg` 的分母、`unmerge` 的回填。
  - 最终统计：`match_indices` 600 组 + `merge_reduce` 510 组 + `size_weighted_merge` 504 组 + `unmerge` 520 组，**全部一致**。
- **判题用例**：4 个 `check_*`（`check_match_indices` / `check_merge_reduce` / `check_size_weighted_merge` / `check_unmerge`），各含独立纯 Python oracle（Fraction 精确余弦）、手算锚点与网格对拍。
- **手算锚点钉住**：max 并列取下标最小、`class_token` 保护行 vs `distill_token` 保护列、`class_token` 才排序 `unm`、`r` 被 `(T - protected) // 2` 截断、`r <= 0` 的 `do_nothing` 特例、负余弦也能被选中、多个 `src` 落同一 `dst` 的 `mean` 分母、`distill_token` 的交错输出布局、`merge_wavg` 的分母是 size 之和、`unmerge` 回填的是合并后的值、`unmerge` 的输出比输入长 `r`。
- **mutation testing**：47 个单点变异，全部 CAUGHT，且**正确解自检先行确认 PASS**。

## 对拍与 mutation 过程中踩到的坑（可复用经验）

1. **官方浮点不能当逐点 oracle**。第一版对拍直接用官方 float64 当期望值，在 `match[382]` 上炸了（`dst_idx[1][3]` 官方给 `3`、自写给 `1`）。诊断后发现根因是**精确余弦完全相等时浮点尾差翻转了 `max` 的选择**，不是实现 bug。**对拍必须按「官方浮点 vs 精确口径」设计成「分歧定位 + 精确差 <= 尾差量级」的判定**，而不是逐点相等。
2. **分歧定位的索引要用对**。第一版定位函数比较的是 `order[pos-1]` 与 `order[pos]` 的 `nmax`（用的是**精确** order 的下标去索引精确 `nmax`），逻辑上根本对不上分歧的那一对。正确做法是**逐位置比较官方 order 与精确 order 在该位置上的两个元素**的精确 `nmax`。
3. **断言方向别写反**。`locate_official_divergence` 返回 `None` 表示「全部分歧都在尾差量级内」（通过），返回一个 dict 才是「超量级、有真 bug」。第一版把两者写反，导致明明通过却报「未能定位分歧点」。
4. **`merge_wavg` 内部调用的 `merge` 闭包会处理 `distill_token`**。第一版把 `size` 的合并当成普通 `cat([unm, dst])`，与官方不符；官方是**同一个闭包**处理 `x` 和 `size`，所以两者都走交错布局。
5. **`unmerge` 的输入是合并后的张量**。第一版对拍把**原始** `x` 传给了官方 `unmerge`，而官方期望长度 `t - r`；同时自写实现把「合并后长度」当成「原始长度」，两处都错。正确关系是 `t = m + r`。
6. **`_size_weighted_merge` 的 `scaled` 少了一维 batch**。第一版在 batch 循环**内部**构造 `scaled`，导致 `_merge_reduce` 拿到的是单批切片，报 `TypeError: 'int' object is not iterable`。必须先把整个 batch 的 `scaled` 造完再传。
7. **`src` 必须互异**。官方 `src_idx` 是 `argsort` 的前 `r` 个，**必然互异**；`dst_idx` 才允许重复（多个 `a` 指向同一个 `b`）。第一版网格让 `src` 可以重复（`rng.choice` 有放回），导致 `unm` 的长度与 `t1 - r` 不符、`unmerge` 的长度算错。网格应用**无放回抽样**选 `src`，`dst` 才随机重复。
8. **手算锚点要自己复核除法**。`merge_wavg` 的锚点连续错了三次（`18/3` 写成 `12/3`、`50/5` 写成 `30/5`、`merged_size` 写成 `[[3],[3],[4]]` 应为 `[[3],[2],[5]]`），都是自检立刻抓出来的——**锚点算错时解是对的，必须改锚点而不是改解**。
9. **真等价变异必须剔除**。`dot-sum-order`（把点积的求和顺序反过来）存活，因为**整数加法满足结合律**，与顺序无关。换成可测形态 `dot-drop-last`（漏算最后一项）。
10. **变异锚点必须与当前代码逐字一致**。`wavg-size-not-merged` 与 `cos-value-none-nan` 两个锚点因缩进/注释文字不匹配而报 `ANCHOR-ERROR`，改锚点文本后正常。

## 不进判题域的边界

ViT 前向与 `timm` patch 的实际接线（`ToMeBlock` 在 attention 与 mlp 之间插入合并）、比例注意力（`attn + size.log()`）、`parse_r` 的多层 `r` 调度、`merge_source` 的来源矩阵、`kth_bipartite_soft_matching` 与 `random_bipartite_soft_matching` 两个变体、GPU 上的实际加速比与精度损失、真实模型的权重，全部不进判题域。本题只保留四段确定性的纯数值账。
