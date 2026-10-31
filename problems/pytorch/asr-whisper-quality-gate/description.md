# Whisper 质量门控与温度回退账

## 题意

Whisper 的 `transcribe` 对每个 30s 窗口按温度序列回退解码，并用三个阈值判定窗口质量：zlib 压缩比（重复循环）、平均 log 概率（太不可信）、no_speech 概率（静音）。本题转写这套 **窗口级回退 + 段级 skip** 的纯账本：每窗给各温度的解码指标（文本/avg_logprob/no_speech_prob），输出压缩比账、选中温度、回退链、silence 覆盖与最终 action。

## 求解

口径题面钉死（与官方 `decode_with_fallback` + 段级 `should_skip` 逐条一致，常数 2.4 / -1.0 / 0.6）：

1. `compression_ratio(text) = len(utf8 字节) / len(zlib.compress(utf8 字节))`（官方 `whisper/utils.py`）；
2. **逐温度回退**（每温度独立判定，顺序即优先级）：
   - `needs_fallback = cr > cr_t`（too repetitive）**或** `avg_logprob < lp_t`（too low）；
   - 若 `no_speech_prob > ns_t` **且** `avg_logprob < lp_t` → `needs_fallback = False`（silence 覆盖，优先级最高）；
   - `not needs_fallback` → 选中当前温度并停止；全部温度用尽 → 选中最后一个（官方行为：返回最后一次结果）；
3. **段级 skip**（用选中温度的指标）：
   - `should_skip = no_speech_prob > ns_t`；
   - 若 `avg_logprob > lp_t`（**严格大于**）→ 豁免，不 skip；
   - `skip` 的窗口 fast-forward 跳过（官方 `seek += segment_size` 语义），否则 `transcribe`。

注意回退与 skip 的两个比较方向相反：回退的 logprob 条件是 `<`，段级豁免是 `>`；恰等于阈值时两边都不触发。

返回每窗 `{"compression_ratios", "selected_temp_index", "fallback_chain", "silence_covered", "action"}`。

## 数据规模

- 1 ≤ len(windows)、1 ≤ len(temperatures) ≤ 10，每窗指标长度恒等于温度数。
- windows 空、窗缺/多键、指标长度不齐、texts 含非 str、指标非数或 NaN/Inf、负温度、阈值 bool/非数 → `ValueError`。
