# Whisper 30s 窗口 seek 推进账

openai/whisper `whisper/transcribe.py` 主循环（word_timestamps=False 主线）：30s 窗
滑动解码，segment 切分与 seek 推进完全由时间戳 token 决定。模型解码剥离——每个
窗口的 token 序列与 no_speech/avg_logprob 诊断值作入参，只考循环账。

## 常量（官方 audio.py / tokenizer）

- N_FRAMES = 3000（30s mel 帧）、HOP_LENGTH = 160、SAMPLE_RATE = 16000
- time_precision = 160/16000 = 0.02 s / token
- timestamp_begin = 50364（timestamp token t 对应时刻 `(t - begin) * 0.02`）
- input_stride = 2（mel 帧 / token 的比例：3000 帧对应 1500 token）

## 输入

```
solve(content_frames, windows, no_speech_threshold=0.6, logprob_threshold=-1.0)
```

- `content_frames`: int，总 mel 帧数
- `windows`: list[dict]，按窗口顺序给出每次循环的解码结果：
  `{"tokens": list[int], "no_speech_prob": float, "avg_logprob": float}`
  （约束：窗口数足够支撑循环走完，即输入保证合法）
- 阈值同官方默认

## 官方循环（逐条对齐 transcribe.py，word_timestamps=False）

```
seek = 0
while seek < content_frames:
    segment_size = min(N_FRAMES, content_frames - seek)   # clip 边界本题不做
    time_offset = seek * 0.02
    若 no_speech_prob > threshold 且 avg_logprob < logprob_threshold:
        segment 丢弃，seek += segment_size（fast-forward）
    ts = [t >= timestamp_begin]
    single_timestamp_ending = ts[-2:] == [False, True]
    consecutive = [i+1 for i 相邻双时间戳]                  # 切分点
    若 consecutive 非空:
        slices = consecutive（+len(tokens) 若 single_timestamp_ending）
        每段 segment(start=offset+(段首-begin)*0.02, end=offset+(段尾-begin)*0.02)
        single_timestamp_ending → seek += segment_size      # 末时间戳后无语音
        否则 → seek += (tokens[last_slice-1] - begin) * input_stride
        # 未完成段（最后时间戳之后的部分）被忽略丢弃
    否则:
        duration = 有末时间戳（≠begin）时 (末-begin)*0.02，否则整窗 30s
        segment(offset, offset+duration)
        seek += segment_size
```

## 输出 dict

```
{
  "segments": list,   # [(start, end)] 秒，float，round 3 位
  "seeks": list,      # 每次循环后的新 seek（帧）
  "final_seek": int,
}
```

## 示例

```python
solve(6000, [
  {"tokens": [50364, 100, 50364+500, 200, 50364+1000], "no_speech_prob": 0.1, "avg_logprob": -0.5},
  {"tokens": [50364+1000, 300, 50364+2500], "no_speech_prob": 0.1, "avg_logprob": -0.5},
])
# 窗1: 连续双时间戳 [TS,ts500,ts1000] 切分点 [3,5]，无末单时间戳
#      segments=[(0.0,10.0)]（ts1000 后未完成段丢弃），seek += (1000)*2 = 2000
# 窗2: 时间戳首尾 → consecutive=[1,3] 单段 (10.0, 20.0)? token 50464→(50464-50364)*0.02=2.0
#      offset=2000*0.02=40 → start=42.0 end=60.0；single? ts[-2:]=[False,True] → seek += 3000
```

## 约束

- token 序列长度 ≥ 2（`ts[-2:]` 可取）；timestamp token ≥ 50364；
  窗口序列保证 while 循环每次都有窗口可用。

写你的实现到 `solve`。
