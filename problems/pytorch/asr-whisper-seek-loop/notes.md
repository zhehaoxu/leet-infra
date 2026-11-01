# Whisper seek 循环账（notes）

回源：openai/whisper `whisper/transcribe.py`（main loop，word_timestamps=False）+
`whisper/audio.py` 常量。

- **双时间精度**：token/时间戳账用 0.02s（HOP 160 / SR 16000），而 time_offset 用
  mel 帧精度 0.01s（seek 是 mel 帧下标）。初版全用 0.02 被随机对拍当场抓出——
  这是本题最核心的口径差。
- **切分只在相邻双时间戳**：`consecutive = where(ts[:-1] & ts[1:]) + 1`——交替
  时间戳（T,F,T,F,...）不触发切分，走 else 整窗分支（末时间戳定 duration）。
  多个固定锚点因此调整过构造。
- **未完成段丢弃**：非 single ending 时 `seek += (tokens[last_slice-1]-begin) ×
  input_stride`，最后时间戳之后的内容被忽略——seek 可跳到窗口中间（非 3000 边界）。
- **no_speech 双门 AND**：`no_speech_prob > 0.6 且 avg_logprob < -1.0` 才跳窗
  （官方注释 no speech AND high decoding failure）。
- **else 分支 duration**：末时间戳存在且 ≠ begin 时用 `(末-begin)×0.02`，否则
  整窗 `segment_size × 0.01`；末时间戳恰为 begin（0s）→ 整窗。
- word_timestamps/hallucination_silence/clip 边界（recovery 分支）未入题。
  tokens 长度 1 时 `ts[-2:]` 官方 torch 语义为整列表（≠[False,True]），锚点覆盖。
