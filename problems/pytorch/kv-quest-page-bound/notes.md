# 解题说明 · Quest 页级上界打分

## 一句话

用「每页每通道的 key 极值」替 query 估一个**注意力上界**，按上界选页；最后一个 page 恒载、不参与竞标。

## 上界为什么成立

对单个通道 \(d\)，把符号乘进 key 之后页内的极值就是

\[
\max_{i\in\text{page}}\bigl(s_d\,k_{i,d}\bigr),\qquad s_d=\operatorname{sign}(q_d)
\]

于是该项的上界是 \(q_d s_d \cdot \max_i(s_d k_{i,d})\)：

- \(q_d>0\Rightarrow s_d=+1\)，上界 \(=q_d\max_i k_{i,d}\)，而 \(q_d k_{i,d}\le\) 它；
- \(q_d\le 0\Rightarrow s_d=-1\)，上界 \(=q_d\min_i k_{i,d}\)，乘的是非正数，不等号照样成立。

逐通道相加即得整页分数，它**不小于**页内任何一条 token 的真实 \(q\cdot k\)——测试里直接拿这个不等式当断言。

写成一个矩阵乘就是：`(q*sign) @ page_max^T`，`page_max` 是 `k*sign` 的页内 amax。注意没有任何 softmax、没有 `1/√d`：这是上界，不是注意力。

## 三个易错口径

1. **零值归负**。上游写的是 `sign = (q > 0) + (~(q > 0)) * -1`，`q_d = 0` 得到 `-1`。用 `torch.sign` 会得到 0，那样该通道的贡献被整段抹掉，选页结果立刻偏。题面把这条钉成硬断言。
2. **末页恒载但不可选**。`scores` 比页数少一列（`[:, :, :cur_num_pages - 1]`），因为最后一页是刚写入的、下次 decode 才轮到它；`selected` 里却永远带着它。所以真正竞标的名额是 `K = min(page_budget, P) - 1`——先把预算按现有页数截断，再减掉恒载的那个。
3. **打包哨兵**。不足一页的位置用输入 dtype 的 `finfo.min` 补齐后再 amax。这个值不会影响上界（真实值一定更大），但它必须是"极小的负数"而不是 0：某通道整页为负时，0 会把 `page_max` 抬成 0，上界就不再是上界。

## 判分点

- `sign` 只有 ±1 两种取值，且零值必须在 `-1`；
- `page_max` 是 `k*sign` 的 amax，不是 `k` 的 amax——页内极值取自哪一端完全由 query 的符号决定；
- `scores` 的列数、`selected` 的长度都跟 `min(page_budget, P)` 对齐；
- `selected` 每行升序、无重复、末尾恒为 `P-1`；
- 同分取更小的页号：上游 `torch.topk` 不定义同分顺序，本题把它钉成确定性约定，测试用「全 1 的 key」造出全同分局面。

## 边界

- `P < 2` 直接报错：只有一页时"排除末页"会让竞标名额变成 0，上游控制器里也有 `assert cur_page_nums > 1`。
- `page_budget >= P` 时名额是 `P-1`，等价于全加载；`page_budget = 1` 时名额是 0，只留末页。
- 单 query（`Lq = 1`）：上界的符号 `sign` 会沿 kv 轴广播，`qo_len > 1` 时每行 query 的符号不同、无法共用一份 `page_max`，所以上游的 CPU 参考实现最后要 `squeeze(1)`。
- 全程 float64 计算：float32 下 `finfo.min` 与真实 key 混算不会出问题，但分数累加对长 `D` 敏感。

## 复杂度

每步 \(O(P\cdot D)\) 的元数据规约加 \(O(H P D)\) 的点积，对比全量 attention 的 \(O(H L_{\text{kv}} D)\)——`L_kv` 越大省得越多。上游把这段做成 CUDA kernel，本题只考打分口径。

## 出处

`mit-han-lab/Quest`（arXiv 2406.10774）的 CPU 参考实现：

- `quest/tests/test_estimate.py::_ref_cpu_estimate` L17：整个打分路径；
- L40 `sign = (q > 0) + (~(q > 0)) * -1`、L41 `max_key = k * sign`、L42 `postive_query = q * sign`；
- L44 `padding_length = page_size - ((kv_len - 1) % page_size + 1)`；
- L63 `page_max_key = max_key.reshape(..., page_size, head_dim).amax(dim=-2)`；
- L72 `approx_attn_paged = torch.matmul(postive_query.float(), page_max_key.transpose(1, 2))`；
- L73 `[:, :, :cur_num_pages - 1]  # remove the last page`、L75 `squeeze(1)`。

预算与恒载口径来自 `quest/utils/controller.py`：L112 `inference_page_budget = min(self._page_budget, cur_page_nums)`、L115 `kv_indptr_for_approx_decode = [0, inference_page_budget - 1]`、L118-119 topk 缓冲区长度 `inference_page_budget - 1`、L137-142 `need_estimate()` 返回 `cur_page_nums > inference_page_budget`。

**本题自己的补充**（上游没写、题面显式钉死）：同分取更小的页号；`page_size`/`page_budget` 走显式入参而不是控制器状态；`k` 直接作入参而不是先跑一遍 attention。

## 相关题目

- `kv-chunk-independent-topk`：对**已经算出来**的分数分块取 top-k，是纯 topk 账；本题考的是分数本身怎么来。
- `kv-snapkv-pooling`：SnapKV 用观测窗内的真实注意力打分，本题用极值上界，完全不算注意力。
- `kv-h2o-heavy-hitters`：按累计注意力驱逐 token，颗粒是 token；本题颗粒是 page 且只做选择不做驱逐。
