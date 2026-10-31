# 解题说明 · Sparse-dLLM 延迟一步的块内驱逐

## 一句话

dLLM 一次解码一整个 block：拿块内 query 的均值给缓存打分，但把当前 block 的 KV 整段剔除（延迟一步），再跨 head 平均、做一次 `max_pool1d(3,1,1)`，按比例取 top-k。

## 为什么「延迟一步」

自回归解码里新 token 的 KV 是刚算出来的、可以直接参与打分；扩散 LLM 的 block 要迭代多轮去噪，当前 block 的 KV 只是**上一轮的中间态**，用它打分等于拿噪声当证据。上游 `CustomCache.filter_cache` 的做法是：本步先不把当前 block 算进候选，等 block 定稿后再进入下一步的打分。于是候选 = 缓存里 block 之前的一段 + block 之后的一段，长度 `L - block_len`。

注意这不是「只去掉最后一行 query」：整个 block 的 `block_len` 个槽位都不可选。因为 block 之外可能还有**后文**（dLLM 的输入是整段已知的），所以候选要前后两段拼回来，而不是像自回归那样只截前缀。

## 打分链路为什么是这个顺序

```
cand     = cat(k[..., :bef, :], k[..., bef+block:, :], dim=-2)
cand     = cand.repeat_interleave(Hq // Hkv, dim=1)      # GQA
avg_q    = q_block.mean(dim=-2)                          # 块内均值，双向
scores   = avg_q @ cand.transpose(-2, -1)                # (B, Hq, S')
heads_avg= scores.mean(dim=1)                            # 先跨 head 平均
pooled   = max_pool1d(heads_avg, kernel=3, stride=1, pad=1)
keep_idx = argsort(-pooled, stable=True)[:, :int(S' * keep_ratio)]
```

- **块内 query 取均值**：block 的 query 地位相同，用均值才能得到一条块级分数；用最后一行会退化成自回归视角。
- **先跨 head 平均再池化**：池化是对「位置」平滑，必须先把 head 维压掉；逐 head 池化再平均会把不同 head 的峰值位置错开，结果不同（测试 `check_heads_averaged_before_pooling` 专门构造了 `[0.5,0,0.5]` 与 `[0.75,0.5,0.75]` 的对比）。
- **`max_pool1d` 而不是平均池化**：窗口内取最大值，让一个高分 key 把相邻位置一起带进 top-k，保证块内连续性；`kernel=3, stride=1, pad=1` 是上游参数。
- **边界是单侧窗口**：PyTorch `max_pool1d` 越界补 \(-\infty\)，位置 0 只和位置 1 比。若像 `avg_pool1d` 那样补 0，全负的一段会被抬到 0，分数全平。
- **GQA 用 `repeat_interleave`**：KV head \(h\) 对应连续的 `Hq/Hkv` 个 query head；用 `repeat` / `tile` 会把「第几个 query head 用哪个 KV head」搞反。

## 与自回归驱逐的本质区别

- **视角**：H2O / SnapKV / TOVA / StreamingLLM 都是「因果过去 + 新 query」，打分只覆盖前缀；本题的块是双向的，候选同时包含 block 前后，且块内 query 全部参与。
- **时效**：自回归方法用刚写入的 KV 立即打分；本题**延迟一步**，当前 block 整段不参与，避免拿未定稿的中间态当证据。
- **保护规则**：自回归方法普遍带 sink（前几个 token）与 recent window（最近若干 token）豁免；Sparse-dLLM 的训练-free 配置里两者都没有，首尾 token 一样可以被丢（测试 `check_block_window_is_excluded` 明确断言 `0` 与 `5` 都不在保留集里）。
- **预算粒度**：按候选总数的比例 `int(len * keep_ratio)` 一次砍掉，不做累计分数、不做 per-layer 预算。

## 与上游的差异 / 为可判做的简化

- **topk 的粒度**：上游 `filter_cache` 对每个 KV head 各做一次 topk（每个 head 的 keep 集合可以不同）；本题先跨**全部** query head 平均成一条 `(B, S')` 分数，再做一次 topk，返回共享的 `keep_idx`。这是本题为可判做的简化，`scores` 仍逐 head 返回，便于核对上游的中间量。
- **候选定位**：上游的缓存已经压紧过，`filter_cache` 直接在「已过滤序列」上工作；本题把完整缓存 `k` 与 `bef_filtered_len` 一起传入，由 `bef_filtered_len` 定位 block，等价于上游的「块前长度」。
- **返回值**：上游只返回保留的索引；本题额外返回 `avg_q` / `scores` / `heads_avg` / `pooled`，让每一步都可判。
- **`keep_num = 0`**：本题允许返回 `(B, 0)` 空索引；上游在这种极端比例下没有显式分支。
- **同分**：上游 `torch.topk` 不定义同分顺序，本题用 `argsort(-pooled, stable=True)` 钉死「取更小候选下标」。
- 输入校验、`block_len` 与 `q_block` 的一致性检查是本题补充。

## 边界

- `bef_filtered_len + block_len > L`：块窗口越界，报错。
- `L == block_len`：块外没有任何候选，报错。
- `keep_ratio` 极小导致 `keep_num = 0`：合法，返回空索引。
- `pool_kernel_size = 1`：池化退化为恒等，`pooled == heads_avg`。
- `H_q == H_kv`：不做复制，直接逐 head 对应。
- 输入只读；`k` 中 block 的位置由 `bef_filtered_len` 决定，不要求 block 在末尾。

## 复杂度

逐 head 点积是 \(O(B H_q S' D)\)，跨 head 平均与池化是 \(O(B H_q S')\)，topk 是 \(O(B S' \log S')\)。相比对整个缓存重新跑 attention，这一步只是 \(O(S' D)\) 的评分开销。

## 出处

`OpenMOSS/Sparse-dLLM`（arXiv **2508.02558**，在仓库 README 中见到编号）：

- `opencompass/models/sparse_dllm/dream/Cache.py::CustomCache.filter_cache`：延迟一步评分（剔除当前 block）、块内 query 均值、`avg_q · K` 后跨 head 均值、`max_pool1d(kernel=3, stride=1, pad=1)`、per-KV-head `topk(keep_num=int(len * keep_ratio))`；
- 本题的 `k` / `q_block` / `bef_filtered_len` / `block_len` 是上述过程的显式入参化。

## 相关题目

- `kv-h2o-heavy-hitters`：因果前缀 + 累计注意力 + sink/recent，按 token 重要性驱逐；本题是双向 block、延迟一步、无保护位。
- `kv-snapkv-pooling`：也用 `avg_pool1d` 平滑观测窗口内的分数，但那是**因果窗口 + 尾部强制保留**；本题是 `max_pool1d`、窗口覆盖 block 两侧、且没有任何强制保留。
- `kv-tova-head-mean`：只看最后一个 query 的 head 均值；本题取块内 query 均值，是双向视角。
- `kv-streaming-llm-sink`：sink + 滑窗的硬保护；本题明确没有 sink / local 规则，驱逐只由分数决定。
