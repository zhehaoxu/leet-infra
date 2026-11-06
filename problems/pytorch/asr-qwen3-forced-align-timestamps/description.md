# Qwen3-ForcedAligner NAR 时间戳账

## 背景

Qwen3-ForcedAligner-0.6B（QwenLM/Qwen3-ASR）是 LALM 式**非自回归**强制对齐器：不做 CTC/DTW 对齐，而是把转写文本按词切开、词间与词尾插入时间戳占位，LLM 一次性前向的 argmax 只在占位位置取值，乘 `timestamp_segment_time` 得到时间戳序列，后处理成每个词的 (start, end)。本题重建这条可判定的后处理链（模型前向不入题，argmax 结果作显式入参）。

## 机制拆解

### 1. 分词（tokenize）

`tokenize_space_lang`：空格分段 → `clean_token`（只保留 Unicode 字母 L\* / 数字 N\* / 撇号 `'`，标点剔除）→ `split_segment_with_chinese`（CJK 逐字切成单字 token、非 CJK 连续段合并），空段丢弃。Chinese/English 都走该分支；Japanese/Korean 走 nagisa / soynlp 外部分词器，判题环境无这些库，不入题。

### 2. 占位编码（encode）

`"<timestamp><timestamp>".join(word_list) + "<timestamp><timestamp>"`，前面拼 `<|audio_start|><|audio_pad|><|audio_end|>`——**每个词恰 2 个时间戳位**（词间 2 个 + 词尾 2 个；空词表也留尾部 2 个）。

### 3. NAR 取值（masked_timestamps）

`logits.argmax(-1)` 后，`output_ids[input_ids == timestamp_token_id]` 取出占位位置的预测 id，再乘 `timestamp_segment_time`。**先掩码再乘**：非占位位的大值不进入结果；取值数恰为 `2 × 词数`。

### 4. LIS 单调修复（fix）

时间戳预测可能非单调，官方用最长**非降**子序列（LIS）挑出正常点：

```
dp[i] = 1；若 data[j] <= data[i] 且 dp[j]+1 > dp[i] → dp[i]=dp[j]+1, parent[i]=j
max_idx = dp.index(max(dp))        # 平分取最早
parent 链回溯 → is_normal 标记（其余为异常点）
```

连续异常段按长度分治：

- 段长 **≤2**：左右最近正常值（从已修复的 `result` 读），按 `(k-(i-1)) <= (j-k)` 距离择一（平分取左）；单侧缺失时取存在侧；
- 段长 **>2**：`step = (right-left)/(anomaly_count+1)` 线性插值（分母含两端点），单侧缺失时平铺；
- 最后 `[int(v) for v in result]` 截断（正浮点向零 = 向下取整）。

### 5. 词 span 映射（parse）

修复后每词取 `[2i]` 作 start、`[2i+1]` 作 end，`round(v / 1000.0, 3)` 转秒（fixed 已 int，`/1000` 恰 3 位小数；round 位数错仍可观测）。

## 判题域

- 文本、input_ids / output_ids、timestamp 序列全部显式入参，fp64；字符串账与整数账为主，无 RNG；
- 5 mode：tokenize、encode、masked_timestamps、fix、parse；
- 不出：LLM 前向与真实权重、日/韩外部分词器、音频 IO 与重采样；与 `asr-whisperx-dtw-align`（CTC emission trellis DP 回溯）、`asr-cif-fire-emit`（发射状态机）划界清晰。
