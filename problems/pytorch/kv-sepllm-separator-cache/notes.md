# 解题说明 · SepLLM 分隔符缓存

## 一句话

一次压缩就是选「sink 前缀 + 分隔符区 + 本地窗」三段并按下标聚集：分隔符区按位置升序取候选，超容量只留最新的 `sep_cache_size` 个，新右边界为 `init + min(分隔符数, s)`。

## 为什么是这套账

三段在槽位空间上连续、且按前提互不重叠，于是压缩退化成纯下标算术——上游 `compress_kv_cache_and_tokids` 也正是按 `[initial_kv, sep_kv, local_kv]` 拼接并同时压紧 token id。

```
right   = init if boundary == 0 else boundary      # first compression: accumulated region is empty
sep_all = list(range(init, right))                 # separators accumulated so far
sep_all += [p for p in range(right, L - local) if ids[p] in sep_set]   # scan the past window
sep_all = sep_all[-cap:]                           # newest separators win
kept    = list(range(init)) + sep_all + list(range(L - local, L))
new_boundary = init + len(sep_all)
```

`sep_all` 天然升序：累积区在前、过去窗口在后，取尾切片就是淘汰位置最小的那批；`kept` 拼完三段就是最终布局，`index_select` 聚集的 KV 与压紧后的 `token_ids` 同序，下一轮 `boundary` 因此直接当压紧后的下标用。

## 三个易错口径

1. **超容量留最新、不是最旧**：上游 `USE_MAX_SEP_CACHE` 分支 `slice(sep_len - max_sep_len, sep_len)` 取的就是尾部（wheel L917-925）；留最旧会让 `kept` 与 `boundary` 同时变。
2. **`boundary = 0` 只在首次压缩等价于空分隔符区**：上游 L886-887 把 `sep_exrange <= 0` 折叠成 `init`，紧接着 L889 断言过去窗口非空；L894 的 `Before_First_Time_Compress_Flag` 跳过 L897 的累积区切片，因为 `slice(init, init)` 会撞上 L1227 的 `start < end`。
3. **sink 与本地窗里的分隔符不占名额**：`[0, init)` 里的分隔符按 sink 保留一次、不计入总数；末尾 `local_size` 个无条件保留；只有过去窗口 `[right, L - local)` 扫 id 出候选。

## 判分点

- `kept`：int64、严格升序、总长 ≤ `init + s + local`，前 `init` 个恰是 sink、末尾 `local` 个恰是本地窗；
- `token_ids` 与逐层 `(H, n, D)` KV 按 `kept` 同序聚集、dtype 不变，测试对 numpy oracle、容差 1e-5（float32）/ 2e-9（float64）；`boundary` 是 Python `int`，等于 `init + min(分隔符总数, s)`，指向压紧后的槽位空间；
- 输入只读；内部 float64 聚集后回输入 dtype；非连续视图与连续副本同结果；非法形状/dtype、非 int 参数、`boundary ∈ (0, init)`、过去窗口为空一律 `ValueError`，不许静默截断。

## 边界

- **过去窗口为空即错**：`L - local_size <= max(boundary, init)`（`boundary = 0` 按 `init` 计）正是上游 L889 的断言，本题转成 `ValueError`，顺带保证三段不重叠。
- **不增也不耗分隔符的边界**：窗口没有新分隔符且区未超容量时 `boundary` 原样不动、累积区整体保留（`check_region_grows_under_cap` 的 quiet 场景）；本地窗全是分隔符也照样整段保留，不占 `sep_cache_size`、不推进边界（`check_sink_and_local_do_not_consume_budget` 的 tail 场景）。
- **流式多轮**：`check_chained_stream` 连续 4 次压缩、每次追加 4 个 token，流够长时边界最终锁在 `init + s`，缓存长不超 `a + s + w`。

## 复杂度

扫过去窗口 \(O(L)\)，聚集是 \(O(\sum_{\text{layers}} H\cdot n\cdot D)\)，\(n\le a+s+w\) 与流长无关；每步额外内存只有压紧后的 KV——训练无关配置 `a=4/s=128/w=256` 合计 388，仍给 `cache_size=512` 留了余量（README L534 要求 `a+s+w<c`）。

## 出处

`HKUDS/SepLLM`（ICML 2025，arXiv 2412.12094）。训练无关实现打包在 `package/transformers-4.38.0.post1+sepllm-py3-none-any.whl`，以下引用均对解包后的 `transformers/cache_utils.py` 逐行核对：

- L337 `class SepCache`；L640-641 `sep_exrange = 0`、`max_sep_exidx = init + sep_cache_size`；L679 llama3 的标点 id 表；
- `compress_kv_cache_and_tokids_layer_wise` L857-940：L886-887 首次压缩折叠、L889 断言、L894 首次标记、L897/899/902 三段切片 `[init, sep_exrange)`、`[sep_exrange, L-w)`、`[L-w, L)`、L917-925 超容量取尾切片并锁右界、L936 按 `[initial_kv, sep_kv, local_kv]` 拼接；
- `compress_past_win_2_seps` L943-992：L945-948 用 `separator_token_ids` 逐位 OR 成布尔掩码、按位置升序保留；`update_kv_cache_and_past_tok_ids` L817 默认 `SEP_ACCUMULATION=True`，与 `USE_MAX_SEP_CACHE=True` 合成本题口径；
- `slice_kv_cache_and_tokids` L1138、L1227 `assert start < end`；Streaming 分支 `Streaming-SepLLM/sepllm_kv_cache/kv_cache_manager.py::compress_kv_cache_and_tokids` L257-321 是同名同构例程；默认参数 `a=4/s=128/w=256/c=512/USE_MAX_SEP_CACHE=True` 出自 `TrainingFree-SepLLM/demo.py` L27；
- 位置偏移属模型侧：wheel 里 `apply_shifted_pos_emb` L995、Streaming 分支 `enable_pos_shifting.py::enable_pos_shifting` L4，本题不做（README L53：training-free 场景不需要 PE shifting）。

**本题自己的补充**：单序列、无 batch（上游 `SEP_PADDING_IN_BATCH` 会批内补齐/截断对齐）；不引入 `left_padding_offset`；`boundary` 与分隔符 id 集合是显式入参而非逐层内部状态；输出显式返回 `kept` 下标并用 `index_select` 聚集，不做空槽复用、也不对 KV 做任何数值合并；上游断言改抛 `ValueError` 并补齐形状/dtype/参数校验，`boundary` 钉成 Python `int`。

## 相关题目

- `kv-streaming-llm-sink`：同样保「sink + 最近窗」，但中间整段扔掉；本题多出一个按**内容**挑选的分隔符区，有独立容量与「最新优先」淘汰。
- `kv-chunkkv-cumulative`：按语义块整块保留、块级打分；本题的分隔符是**散点**不是连续块，淘汰只看位置。
- `kv-h2o-heavy-hitters`：按累计注意力驱逐；本题完全不看注意力，只看 token id 与位置。
