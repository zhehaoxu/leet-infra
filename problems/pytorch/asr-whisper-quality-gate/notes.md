# notes — asr-whisper-quality-gate（order 671）

## 回源

- 出处：`openai/whisper` `whisper/transcribe.py` 的 `decode_with_fallback` 与段级 `should_skip`，`whisper/utils.py` 的 `compression_ratio`（zlib，非 gzip 模块）。
- 三个官方常数与语义：`compression_ratio_threshold=2.4`（> 才算 too repetitive）、`logprob_threshold=-1.0`（< 才算 too low）、`no_speech_threshold=0.6`。
- 两个易错点：
  - **silence 覆盖优先级最高**：`no_speech_prob > 0.6 且 avg_logprob < -1.0` 时即使前面判了 needs_fallback=True 也要翻回 False（即"确认是静音，别重试了"），且此时温度 0 就通过、不回退；
  - **段级豁免方向相反**：skip 需要 `no_speech_prob > 0.6` 且**不被** `avg_logprob > -1.0` 豁免；恰等阈值两边都不触发（严格不等式）。
- 与 #109（TTS）划界：本题是**解码后质量账**；与候选 ⑨ VAD 划界：VAD 是前段切分。

## 判题面

- solve 入参：windows（texts/avg_logprob/no_speech_prob 各按温度一列）、temperatures、三阈值。返回逐窗账本。
- oracle：`test_cases._py` 独立实现（while 逐温度推进、covered 标志另一条路径），120 组随机对拍（温度数 1–4、一半概率造重复文本、边界值 0.59/0.6/0.61 与 -1.0 混入）。
- 锚点：温度 0 直接通过；cr 高回退（"abc"*20 cr≈4.29）；silence 覆盖（每温度 nsp=0.9/alp=-2 → 温度 0 通过但 action=skip）；nsp 高但 alp 高豁免；全部温度用尽选最后一个（官方无 fail 概念）；cr/alp/nsp 三个恰等边界；空文本 cr<1（zlib 头开销）；多字节按 utf-8 字节数。
- 非法输入：windows/temperatures 空、窗缺/多键、指标长度≠温度数、texts 非字符串、指标 NaN/Inf/bool、负温度、阈值 bool → ValueError。

## 变异记录（9 条全抓）

cr-threshold-gte（> → >=）、logprob-lte、silence-no-logprob-cond（覆盖丢 logprob 条件）、no-silence-override（覆盖改加深）、skip-exempt-inverted（豁免反号）、skip-gte、no-break-on-accept（通过后不 break 恒用最后温度）、cr-inverted（压缩比反比）、exhausted-pick-first（用尽后选温度 0）。

## 红线

模型调用/解码不出（指标全显式入参）；真音频 IO 不出；与 `audio` 前端合集（信号处理）划界：本题是**解码后质量账**。
