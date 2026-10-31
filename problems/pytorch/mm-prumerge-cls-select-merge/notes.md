# 解析：LLaVA-PruMerge 的 CLS 选 token 与余弦残差合并

## 实现思路

先把参数全部校验干净：`reduction_ratio` 必须是 `(0,1]` 内的 Python 实数，`eps` 必须是正实数，`int(N * reduction_ratio)` 为 `0` 时立即抛 `ValueError`，因为「一个 token 都不保留」在本任务里没有定义。

选择用一次 `descending=True` 且 `stable=True` 的排序完成，并列的注意力值因此保留原始升序下标；取前 `k` 个之后再排一次序，得到升序的 `kept_indexs`。`pruned_indexs` 用一张布尔掩码配 `argsort(..., stable=True)` 得到：未选中的 `0` 排在前面且保持升序，切前 `N-k` 列即为被合并掉的 patch。两条索引都是升序，两者不重叠且并起来是完整的 `0..N-1`。

`F.normalize(p=2, dim=-1, eps=eps)` 的分母就是 `max(||x||, eps)`，与契约里的兜底完全等价，所以零向量的单位向量是全零而不是 NaN。相似度用 `bmm`，簇心用 `argmax`（并列自动取第一个最大值），概率用 `softmax` 沿整个 kept 维，随后 `gather` 出簇心那一列再乘 `C ** -0.5`。合并把 `kept` 的拷贝作为初值，用 `scatter_add` 沿簇心列累加 `weight * pruned` 行：这恰好实现「自己的特征 + 所有分配到的加权邻居」，并且没有被分配的列天然保持原值。`N-k = 0` 时所有中间量都是空的，`merged` 直接等于按升序排列的输入 patch。

## 与官方实现的刻意差异

官方 `llava/model/multimodal_encoder/clip_encoder.py` 的片段是本次规则的依据，以下差异都是刻意固定下来的，不是随手简化：

- **顺序**。`torch.topk` 给出的保留下标是注意力**降序**，官方片段直接在 `idx` 上做后续计算。本题把 kept 下标重新排成**升序**，`merged[j]` 与 `kept_indexs[j]` 因此同序对齐，判题也只依赖一条确定的顺序。
- **并列**。`topk` 与 `argmax` 在相等取值上的取舍是后端相关的。本题显式规定：选择时「值降序、下标升序」，簇心时「第一个最大值」，样例里零向量的全零余弦行就因此落到第 0 列。
- **索引输出的 batch 轴**。官方片段里保留集合是按行算的（`idx` 逐行不同），本题额外把 `kept_indexs`、`pruned_indexs` 连同 `cluster`、`weights`、`scores` 一起返回，全部带 batch 维 `[B,k]` / `[B,N-k]`。这是练习新增的可观察量，不是上游的返回值声明。
- **`int()` 截断保留**。官方的 `int(N * reduction_ratio)` 是截断而非四舍五入，本题照此实现并在契约里写明，避免与 FastV 的 `round()` 混淆。
- **softmax 之后再缩放保留**。官方写的是 `F.softmax(attn, dim=-1) * C ** -0.5`：概率先在**整个 kept 集合**上归一化，再取簇心那一项，最后乘通道数的平方根倒数；簇内不再归一化。本题完整保留这一非直观语义，并用测试断言一个簇的权重和不为 `1`。
- **eps 兜底**。官方依赖 `F.normalize` 的默认 `eps`；本题把 `eps` 暴露成参数并规定 `norm_j = max(||x||_2, eps)`，让零模长 patch 的行为可判定。

## 独立验证

oracle 不复用任何 torch 的选择或归约。它把输入 `tolist()` 成嵌套列表，用 `math.sqrt` 与标量循环手写点积和模长，用 `max(norm, eps)` 兜底，用严格大于的循环手写 `argmax`（并列自然取第一个），用手写减最大值、`math.exp` 与求和得到 softmax 概率，再用 Python 列表逐个簇累加残差。随机用例与该 oracle 逐元素对拍，三组索引与簇分配要求 `torch.equal` 精确一致。

16 个 `check_*` 覆盖两个实跑样例、截断而非四舍五入的边界表（`int(5*0.5)=2`、`int(4*0.25)=1`、`int(3*0.5)=1`、`int(3*0.99)=2`）、全部注意力并列、`reduction_ratio=1` 时 pruned 为空且 `merged` 逐字节等于输入、一个簇两个成员时权重和不为 `1`、`C=1/4/9` 下权重恰为 `p * C**-0.5`、零模长 patch 无 NaN 且多种 `eps` 结果一致、`argmax` 并列取第一个、空簇原样返回、残差等于「自身 + 分配到项之和」、`scores` 等于最大余弦且对 patch 幅度缩放不变、float32/float64、非连续视图、输入不变性、十条随机形状扫描，以及全部非法输入（比例越界或为非实数、`k` 截断为 `0`、`eps<=0`、形状不匹配、空 batch/空 patch/零特征维、dtype 不一致、非有限值等）。

## 复杂度

设 `k = int(N * reduction_ratio)`。选择为 `O(B N log N)`，余弦相似度为 `O(B (N-k) k C)`，softmax 与 gather 为 `O(B (N-k) k)`，残差累加为 `O(B N C)`，合计 `O(B (N log N + (N-k) k C + N C))`。除输出外的主要中间量是 `(B, N-k, k)` 的相似度矩阵与 `(B, N, C)` 的单位向量，即 `O(B (N-k) k C)` 级别的特征副本。`k` 约为 `N/8` 时，相似度矩阵本身远小于返回的特征副本。

## 来源与范围

依据已回源的 LLaVA-PruMerge 官方仓库 `42Shawn/LLaVA-PruMerge`，文件 `llava/model/multimodal_encoder/clip_encoder.py`：CLS 行对 patch 列的注意力、`torch.topk(cls_attn, int(N * reduction_ratio), dim=1, largest=True)` 的选择、`[c for c in range(N) if c not in row_idx]` 的补集、`F.normalize` 后的 `bmm` 余弦、`torch.argmax(cos_sim_matrix, dim=2)` 的簇心、`F.softmax(attn, dim=-1) * C ** -0.5` 的权重，以及 `updated_center = x_others[b, i, :] + weighted_avg` 的残差合并，都取自这段实现；视觉 token 的缩减确实发生在视觉塔内部、投影层之前，默认保留约 `1/8` 的 patch。

刻意不在范围内：论文用「四分位距 + IQR 异常值规则」在图与图之间自适应决定该保留多少 token，本题把数量固定成 `int(N * reduction_ratio)` 这一个参数；逐层的 LLM 侧剪枝变体；把 CLS 与最后一个 token 拼接成 `pooled` 输出的做法；以及真正的 CLIP 前向——注意力矩阵与 patch 特征都是显式输入，调用方负责保证 `cls_attn` 确实是已对 head 平均过的 CLS→patch 注意力。本题是「选择 + 合并」这两步的确定性复刻，不是完整的 LLaVA-PruMerge 模型。

```text
.venv-backend/bin/python scripts/add_problem.py --selftest mm-prumerge-cls-select-merge --mode submit
```
