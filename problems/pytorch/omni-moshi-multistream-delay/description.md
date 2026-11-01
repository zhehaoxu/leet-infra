# Moshi 多流 acoustic delay 与覆盖链账

## 题意

kyutai-labs/moshi（rust 版 `lm_generate_multistream.rs`，Config::v0_1）的 Mimi 双流
账：每步 16 个音频 codebook——左半 8 个生成流（depformer），右半 8 个输入流（用户
语音）。模型采样剥离（depformer 输出作入参），只考 token 矩阵与播放账。

常量：generated=8、input=8、vocab=2049（audio_pad=2048=vocab-1）、acoustic_delay=2。

每步（源码顺序）：
1. 写输入半：`audio_tokens[step][8+c] = input[c]`（不足步不写）。
2. 写生成半：`for c in 0..8`：delay = 0 若 c==0 否则 2；目标槽
   `max(step-delay, 0)`（saturating_sub——step 0..2 时 cb1-7 反复写 slot0，
   **后写覆盖先写**，官方注释确认）；None 输出写 audio_pad。**cb8 不在写循环**
   （cb8 只是读侧零延迟 codebook）。
3. 播放判定 `last_audio_tokens()`：`step <= 2` → None；否则读帧
   `audio_tokens[step-3]`；帧内任一值 >= 2048 → None（整帧不播），否则 16 值可播。

矩阵尺寸 `(max_steps + 2) × 16`，初始 UNGENERATED（-1）。

## 接口

```python
solve(max_steps, input_frames, dep_outputs)
```

返回：

```python
{
  "audio_tokens": list,    # 完整矩阵（含尾部 delay 行）
  "playable": list,        # 每步可播帧 list 或 None
  "ungenerated_left": int, # 残留 UNGENERATED 格数
}
```

约束：max_steps ≥ 1；input 值 < 2048；dep 输出值 < 2048 或 None。

## 示例

```python
solve(6, [[100+i for i in range(8)]]*6,
      [[10+step*8+c for c in range(8)] for step in range(6)])
# slot0 = [10, 27..33]（cb0=step0 输出、cb1-7 被 step2 覆盖）
# slot1 = [18, 35..41]；slot2 = [26, 43..49]
# playable[0..2]=None；playable[3]=slot0+输入帧（可播）
```

## 数据规模与判题

- max_steps ≤ 10；300 组随机对拍；变异 13 条全抓（cb0 延迟翻转、saturating 移除、
  写循环含 cb8、None 写 0、输入半丢写/清零、播放窗口边界、帧偏移、any→all、
  阈值 >、矩阵尾行）。`list(frame)`→`frame` 为数学等价变异（slot 记录后无后续写
  命中，引用与拷贝不可区分）——不收录。
