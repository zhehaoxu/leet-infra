# Moshi 多流 acoustic delay 与覆盖链账

kyutai-labs/moshi（rust 版 `rust/moshi-core/src/lm_generate_multistream.rs`，Config
v0_1）的 Mimi 双流账：每步同时处理 16 个音频 codebook——左半 8 个是**生成**流
（depformer 采样），右半 8 个是**输入**流（用户语音）。生成与播放之间有
acoustic_delay=2 的滞后，且前几帧存在官方注释明说的"后写覆盖先写"。本题剥离
模型采样（depformer 每步输出与文本 token 均作入参），只考 token 矩阵与播放账。

## 常量（Config::v0_1，全部入参显式给定）

- generated_codebooks=8, input_codebooks=8（合计 16 列）
- audio_vocab_size=2049 → audio_pad_token = 2048（= vocab-1）
- acoustic_delay = 2

## 输入

```
solve(max_steps, input_frames, dep_outputs)
```

- `max_steps`: int，步数（步号 0..max_steps-1）
- `input_frames`: list[list[int]]，每步 8 个输入 codebook 值（右半 cb8..15；
  不足 max_steps 的步不写）
- `dep_outputs`: list[list[int] | None]，每步 depformer 采样输出（8 个值）或
  None（表示未生成 → 全部写 audio_pad）

## 官方口径（逐条对齐 lm_generate_multistream.rs 的 step_）

1. **矩阵尺寸**：`audio_tokens` 为 `(max_steps + acoustic_delay) × 16`，初始全部
   UNGENERATED（本题用 -1 表示）。
2. **写入顺序**：每步先写输入半（`audio_tokens[step][8+c] = input[c]`），再写
   生成半。
3. **生成半写入**：`for c in 0..8`：
   - delay = 0 若 c == 0，否则 acoustic_delay（**cb8 不在写入循环内**——8 只是
     读侧的零延迟 codebook，不是写侧）；
   - 目标槽 `max(step - delay, 0)`（saturating_sub：step=0,1 时 cb1-7 都写 slot 0，
     **后写覆盖先写**——官方注释确认前几帧必然发生）；
   - 写入值 = dep_outputs[step][c]，dep_outputs[step] 为 None 时写 audio_pad。
4. **播放判定**（`last_audio_tokens()`，每步之后调用）：
   - `step <= acoustic_delay` → None（前 3 步无可播帧）；
   - 否则取帧 `audio_tokens[step - delay - 1]`（step 领先 1 + 2 帧延迟）；
   - 帧内**任一** codebook 值 >= audio_vocab_size - 1（即 audio_pad）→ None
     （整帧不播）；否则该帧 16 值可播。

## 输出 dict

```
{
  "audio_tokens": list,   # 完整矩阵（含尾部 delay 行的 UNGENERATED）
  "playable": list,       # 每步 last_audio_tokens()：帧 list 或 None
  "ungenerated_left": int,# 矩阵中残留 UNGENERATED 的格数（生成半尾部未写满）
}
```

## 示例

```python
solve(6, [[100+i for i in range(8)]]*6,
      [[10+step*8+c for c in range(8)] for step in range(6)])
# slot0 = [10, 27..33]（cb0=step0 输出、cb1-7 被 step2 的写覆盖）
# slot1 = [18, 35..41]，slot2 = [26, 43..49]
# playable[3] = slot0 + 输入帧 = [10,27..33,100..107]，playable[0..2] = None
```

## 约束

- dep_outputs 长度 ≤ max_steps；input 值 < 2048；dep 输出值 < 2048；
  max_steps ≥ 1。

写你的实现到 `solve`。
