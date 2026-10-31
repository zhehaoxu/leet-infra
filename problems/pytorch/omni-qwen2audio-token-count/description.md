# Qwen2-Audio：音频占位 token 展开数账

## 背景与位置

Qwen2-Audio 的对话序列里音频只占**一个** `<|AUDIO|>` 占位 token，模型 forward 时按音频编码器实际输出长度把它展开成 `audio_len` 个音频 token。HF transformers `processing_qwen2_audio.py` + `modeling_qwen2_audio.py` 的账本：`feature_attention_mask` 记录 mel 批内 padding 前的真实长度，`_get_feat_extract_output_lengths` 给出两层 conv 后的输出帧数。

## 精确账目

1. **池化消 padding**：`pooled_l = Σ feature_attention_mask[i]`（行内 0/1 求和），padding 长度对结果零影响——同一音频 pad 到 400 或 100 帧结果一致。
2. **两层 conv 输出长度**：`audio_len = ((pooled_l − 1)//2 + 1 − 2)//2 + 1`；`pooled_l ≤ 2` 时 `audio_len = 0`（音频被压没，占位替换成空、序列缩短）。边界：L=1,2→0；L=3..6→1；L=7..10→2。
3. **占位计数守恒**：`input_ids` 中音频占位 token 数必须等于音频条数——这是 processor 组 batch 的硬约束。
4. **展开后长度守恒**：`expanded_len = len(input_ids) − 占位数 + Σ audio_len`（每个占位被替换，audio_len=0 时净删 1 个 token）。

## 约束

1. `input_ids` 只允许 0（文本）/ 2（音频占位）；mask 值只允许 0/1。
2. 每条音频 pooled ≥ 1。
3. 多条音频按**音频条目序**（mask 行序）对应 audio_len，与占位在 `input_ids` 中的出现位置无关。
4. 禁改入参；全整型账。

## 数据规模

- 音频数 ≤ 4，单音频 mel ≤ 10⁴，序列长 ≤ 512。
- `expanded_len` ≤ 10⁴。

## 验收锚点（节选）

- 200 帧音频 + 56 pad、序列 `[T,T,A,T,T]` → pooled 200、audio_len 50、expanded 54。
- mel=2 的音频 → audio_len 0、expanded_len = 原 len − 1。
- 双音频 `[A,T,A]`、mask [200 帧, 8 帧] → [50, 2]、expanded = 3−2+52。
