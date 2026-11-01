# Whisper 30s 窗口 seek 推进账

## 题意

openai/whisper `whisper/transcribe.py` 主循环（word_timestamps=False 主线）：30s 窗
滑动解码，segment 切分与 seek 推进完全由时间戳 token 决定。模型解码剥离——每窗口
的 token 序列与 no_speech/avg_logprob 诊断值作入参。

常量：N_FRAMES=3000（30s mel 帧）、time_precision=0.02s/token（HOP 160/SR 16000）、
timestamp_begin=50364、input_stride=2、**time_offset = seek × HOP/SR = seek × 0.01**
（mel 帧精度——与 token 精度 0.02 不同，双精度是最隐蔽账点）。

每轮循环（官方顺序）：
1. `segment_size = min(3000, content_frames - seek)`；
2. `no_speech_prob > 0.6` **且** `avg_logprob < -1.0` → 段丢弃、seek += segment_size；
3. `ts = tokens >= 50364`；`single_timestamp_ending = ts[-2:] == [False, True]`；
4. `consecutive` = 相邻双时间戳下标 +1（**只有相邻对才切分**——交替时间戳不切）；
   有 consecutive → 按 slice 切段（start/end = 段首/尾 token - begin × 0.02 +
   offset）；`single_timestamp_ending` → seek += segment_size；否则
   `seek += (tokens[last_slice-1] - begin) × input_stride`（**未完成段丢弃**，seek
   跳到末时间戳）；
5. 无 consecutive → 有末时间戳（≠begin）时 duration=(末-begin)×0.02 否则整窗
   （segment_size × 0.01）；seek += segment_size。

## 接口

```python
solve(content_frames, windows, no_speech_threshold=0.6, logprob_threshold=-1.0)
# windows: [{"tokens": [...], "no_speech_prob": f, "avg_logprob": f}]（按循环顺序）
```

返回：

```python
{"segments": [(start, end)], "seeks": [...], "final_seek": int}
```

## 示例

```python
solve(6000, [
  {"tokens": [50364, 100, 50364+500, 50364+1000], "no_speech_prob": 0.1, "avg_logprob": -0.5},
  {"tokens": [50364, 300, 50364+1500], "no_speech_prob": 0.1, "avg_logprob": -0.5},
  {"tokens": [50364, 9, 50364+1500], "no_speech_prob": 0.1, "avg_logprob": -0.5},
])
# 相邻对 index2,3 → 段1 (0,10)；未完成段丢弃 → seek += 500*2 = 1000
# 窗2 offset=10s single ending → (10,40) seek=4000；窗3 → (40,70) seek=6000
```

## 数据规模与判题

- 窗口序列由输入保证足够；300 组随机对拍；变异 13 条全抓（min 裁剪、no_speech
  AND→OR/单门、`>=`→`>`、single 判定、consecutive 下标、AND→OR、段 end 用首
  token、未完成 seek 去 stride、else 末时间戳门、duration/offset 精度 0.01↔0.02）。
