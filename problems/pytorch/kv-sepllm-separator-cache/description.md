# KV Cache：SepLLM 分隔符 token 的流式保留

## 题目描述

SepLLM（ICML 2025）的观察是：标点、换行这类**分隔符 token** 吃掉了远超其数量的注意力。训练无关的 KV 压缩因此只留三段——开头几个 sink、流里见过的分隔符、最近一段窗口——中间整段扔掉，缓存上界是 `init_cache_size + sep_cache_size + local_size`。

三段在槽位上是连续的：

1. **sink 前缀** `[0, init_cache_size)`，永久保留；
2. **分隔符区** `[init_cache_size, sep_exrange)`，攒着流里见过的分隔符 KV；
3. **本地窗** `[L - local_size, L)`，最近 `local_size` 个 token。

压缩一次：把分隔符区右边到本地窗左边的「过去窗口」扫一遍，挑出 id 落在 `separator_token_ids` 里的位置接在分隔符区尾部；攒下的分隔符一旦超过 `sep_cache_size`，**只留位置最大的那些**，右边界随之锁在 `init_cache_size + sep_cache_size`。上游 `compress_kv_cache_and_tokids` 同时吐出聚集后的 KV 与**压紧后的 token id**，两者同序：sink → 分隔符（位置升序）→ 本地窗。

## 任务

实现 `solve(token_ids, sep_ids, keys, values, init_cache_size=4, sep_cache_size=32, local_size=224, boundary=0)`：

- `token_ids`：`(L,)` int64，当前 cache 每个槽位上的 token id；
- `sep_ids`：分隔符 id 集合，list / tuple / set / int64 tensor 均可；
- `keys`、`values`：逐层 `(H, L, D)` 张量列表，层数一致、形状与 dtype（float32/float64）一致；
- `init_cache_size` / `sep_cache_size` / `local_size`：正整数；`boundary`：当前分隔符区右边界，`0` 表示尚未压缩过。

返回 5 键字典：

- `kept`：`(n,)` int64，保留槽位的**原始下标**，升序；
- `token_ids`：`(n,)` int64，压紧后的 id；
- `keys`、`values`：逐层 `(H, n, D)`，沿 seq 维按 `kept` 聚集；
- `boundary`：int，新右边界 `init_cache_size + min(分隔符总数, sep_cache_size)`。

非法输入一律抛 `ValueError`，其中包括上游那行 `assert seq_len - local_size > sep_exrange`：`L - local_size <= max(boundary, init_cache_size)` 时过去窗口为空（首次压缩 `boundary = 0` 按 `init_cache_size` 计），上游直接断言失败。

## 示例

```
token_ids       = [9, 9, 1, 5, 1, 2, 3, 7, 1]   # L = 9, positions 0..8
sep_ids         = {1, 2}
init_cache_size = 2, sep_cache_size = 2, local_size = 3, boundary = 0

sink     = [0, 2)              -> 9, 9
acc      = []                  # boundary = 0: first compression, separator region empty
past_win = [2, 9-3) = [2, 6)   -> 1, 5, 1, 2     separators at 2 (id 1), 4 (id 1), 5 (id 2)
local    = [9-3, 9) = [6, 9)   -> 3, 7, 1

sep_all  = [2, 4, 5]           # 3 candidates for sep_cache_size = 2 slots
sep_keep = [4, 5]              # newest two survive, position 2 is thrown away
boundary = init_cache_size + 2 = 4

kept      = [0, 1, 4, 5, 6, 7, 8]
token_ids = [9, 9, 1, 2, 3, 7, 1]   # the 1 at position 8 rides along as local, not as a sep
gathered  = [k[:, kept, :] for k in keys]      # every layer becomes (H, 7, D)
```

位置 2 是分隔符，却因为「只留最新」被淘汰；位置 8 的 `1` 留在 cache 里是因为它在本地窗，不占分隔符名额。

## 约束

- **最新的分隔符胜出**：候选按位置升序拼接（累积区在前、过去窗口在后），超容量时从尾部切 `sep_cache_size` 个，等价于淘汰位置最小的那些。反过来留最旧的，`kept` 与 `boundary` 会同时变。
- **sink 先占位，不与分隔符区重叠**：`[0, init_cache_size)` 里的分隔符按 sink 保留一次，既不重复出现，也不计入分隔符总数、不推进边界。
- **首次压缩等价于空累积区**：`boundary = 0` 与 `boundary = init_cache_size` 必须同结果；上游靠 `Before_First_Time_Compress_Flag` 跳过一次 `slice(init, init)`，那个切片会撞上 `start < end` 断言。
- **本地窗不参与淘汰**：最后 `local_size` 个 token 无条件保留，哪怕里面全是分隔符。
- **边界是压紧后的下标**：`boundary` 指向 `kept` / `token_ids` 这套新槽位，不是原始位置。
- **越界即报错**：过去窗口必须非空（`L - local_size > max(boundary, init_cache_size)`，首次压缩 `boundary = 0` 按 `init_cache_size` 计），`boundary` 只能为 `0` 或 `>= init_cache_size`。
- **不做位置偏移**：上游 `enable_pos_shifting` 是模型侧的 RoPE 重打补丁，本题只判缓存记账。
