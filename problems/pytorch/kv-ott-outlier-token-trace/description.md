# KV Cache：OTT 低幅值 token 的追踪与回填

## 题目描述

OTT（*Accurate KV Cache Quantization with Outlier Tokens Tracing*，ACL 2025）改的是 KIVI 那套 2-bit KV 量化：KIVI 按「近因」把最新的 `residual_length` 个 token 留在 fp16，OTT 另外**追着一小撮 token 不放**——不是注意力最大的，而是 **|K| 幅值最小**的那几个。这些 token 一旦被塞进 2-bit 组里，组内 min-max 会被它们拉得极窄，所以 OTT 把它们的全精度 K/V 单独存在追踪集里，量化流里只留一个公共代表值。

追踪集是**跨量化步累积**的：`sink_max_size` 是总条数上限，`sink_num` 是每一步最多新增几条。每一步先算一遍 fp 缓冲区里「够整组」的那段 token 的幅值分，再拿它们和**已追踪集合里第 `sink_num` 小的分数**比：只有严格更小的才进来。进来之后整张追踪表按分数重排，于是旧条目会被新条目夹到中间——`sink_id` 不是追加序，而是升序序。

量化本体（`triton_quantize_and_pack_along_last_dim`、group-wise min-max）与 `quant-kivi-kv-asym` 重复，本题不考；考的是**可量化长度取整、追踪判定与整表重排、以及量化流里的均值回填**这三段纯记账。

## 任务

实现 `solve(keys, values, quantized_len, sink_id, sink_scores, group_size=128, residual_length=128, sink_num=5, sink_max_size=32)`：

- `keys`、`values`：同形状同 dtype 的 float32/float64 `(1, H, F, D)`，本层**尚未量化**的 fp 缓冲区（batch 恒为 1，`H` 是 KV head 数）；
- `quantized_len`：此前已量化掉的 token 数，即本次 chunk 的绝对下标偏移；
- `sink_id`：int64 `(n,)`，已追踪 token 的绝对下标；`sink_scores`：与 `keys` 同 dtype 的 `(n,)`，对应的幅值分，已升序；`n == 0` 表示还没追过；
- `group_size` / `residual_length` / `sink_num` / `sink_max_size`：正整数（上游 `example.py` 默认 128 / 128 / 5 / 32）。

规则：`quant_len = max(0, (F - residual_length) // group_size * group_size)`，chunk 取缓冲区前 `quant_len` 个 token；`scores[t]` 是 chunk 里第 `t` 个 token 在**所有 head 与所有通道**上 `|K|` 的均值（value 不参与）；`sink_id` 空时取最小的 `sink_num` 个，非空时门槛取 `sink_scores[sink_num - 1]`、候选是 `scores <` 门槛的位置、且 `n >= sink_max_size` 时整步不新增、否则在候选里取最小的 `min(sink_num, 候选数)` 个；新旧拼接后按分数**稳定升序**重排；最后把本次新增的那批 token 在量化流里替换成「它们跨 token 的逐 head 均值」。

返回 7 键字典：`quant_len`（int64 零维）、`scores` `(quant_len,)`、`new_indices`（int64，chunk 内下标、按分数升序）、`sink_id`（int64，绝对下标、按 `sink_scores` 升序）、`sink_scores`、`keys_out` 与 `values_out`（`(1, H, quant_len, D)`，与输入同 dtype）。

## 示例

```
setup: H = 2, D = 2, group_size = 2, residual_length = 1, sink_num = 2, sink_max_size = 5
       keys[0, 0, t, :] = a[t] * 1.0,  keys[0, 1, t, :] = a[t] * 3.0,  values = keys / 2
       -> scores[t] = mean(|K| over heads and channels) = 2 * a[t]

step 1: F = 5, quantized_len = 0, sink_id = [], sink_scores = []
  a         = [2.0, 0.5, 1.5, 1.0, 3.5]
  quant_len = (5 - 1) // 2 * 2 = 4                 # the 5th token stays in the residual
  scores    = [4.0, 1.0, 3.0, 2.0]                 # chunk positions 0..3
  first step, no threshold: the 2 smallest are positions [1, 3]
  new_indices = [1, 3]   sink_id = [1, 3]   sink_scores = [1.0, 2.0]
  keys_out   = [[[2.0, 2.0], [0.75, 0.75],   [1.5, 1.5],   [0.75, 0.75]],
                [[6.0, 6.0], [2.25, 2.25],   [4.5, 4.5],   [2.25, 2.25]]]
  values_out = [[[1.0, 1.0], [0.375, 0.375], [0.75, 0.75], [0.375, 0.375]],
                [[3.0, 3.0], [1.125, 1.125], [2.25, 2.25], [1.125, 1.125]]]
                 # both traced tokens share one per-head mean: (0.5 + 1.0) / 2 = 0.75

step 2: F = 8, quantized_len = 4, sink_id = [1, 3], sink_scores = [1.0, 2.0]
  a         = [2.5, 0.25, 4.0, 0.75, 3.0, 1.25, 3.5, 1.75]
  quant_len = (8 - 1) // 2 * 2 = 6                 # chunk covers absolute ids 4..9
  scores    = [5.0, 0.5, 8.0, 1.5, 6.0, 2.5]
  threshold = sink_scores[sink_num - 1] = sink_scores[1] = 2.0
  candidates (scores < 2.0) = positions [1, 3];   tracked count 2 < sink_max_size
  new_indices = [1, 3]  ->  absolute ids [1 + 4, 3 + 4] = [5, 7]  with scores [0.5, 1.5]
  merged scores [1.0, 2.0, 0.5, 1.5] -> sorted [0.5, 1.0, 1.5, 2.0]
  sink_id     = [5, 1, 7, 3]                       # old entries are interleaved, not appended
  sink_scores = [0.5, 1.0, 1.5, 2.0]
  keys_out[0, 0, :, 0] = [2.5, 0.5, 4.0, 0.5, 3.0, 1.25]   # (0.25 + 0.75) / 2 = 0.5
```

## 约束

- **门槛是配置的 `sink_num - 1`，不是 `n - 1`**：`sink_scores[sink_num - 1]` 取的是「已追踪集合里第 `sink_num` 小」的分数；追到 5 条以后仍然用第 2 小的那条当门槛。
- **严格小于**：`scores == threshold` 的位置不入选。
- **上限卡的是进入本次之前**：判据是 `n < sink_max_size`，所以一步最多能把追踪集顶到 `sink_max_size + sink_num - 1` 条；已经满额时本次整步不新增，但 `scores` 与 `keys_out` 仍照常返回（没有回填）。
- **取整方向**：`quant_len` 向下取整到 `group_size` 的整数倍，并保证至少留 `residual_length` 个最新 token；`F < residual_length + group_size` 时 `quant_len == 0`，等价于上游 L361 那道门槛没过——本次不量化也不追踪，状态原样回吐，`sink_num` 也不再和 `quant_len` 比大小。
- **回填跨 token 而非跨 head**：取均值的是「本次新增的那批 token」，逐 head 各算一个代表向量，再广播写回这批 token 的全部位置；因为广播的是同一个向量，这批下标用分数序还是升序去取都同结果（参考解按升序取数）。
- **分数与 value 无关**：上游 `metric(k, v)` 签名里带着 `v` 却根本不用，`scores` 只由 K 决定。
- **非法输入抛 ValueError**：`keys`/`values` 不是四维、不是 float32/float64、形状或 dtype 不一致、batch 不等于 1；`sink_id`/`sink_scores` 不是一维、`sink_id` 不是 int64、`sink_scores` dtype 与 `keys` 不一致、两者长度不一致；`sink_id` 非空却少于 `sink_num` 条（上游那行会 `IndexError`）；整数参数不是 int（`bool` 也算）或 `quantized_len`/`residual_length` 为负、`group_size`/`sink_num`/`sink_max_size` 小于 1；`quant_len > 0` 时 `sink_num > quant_len`。
