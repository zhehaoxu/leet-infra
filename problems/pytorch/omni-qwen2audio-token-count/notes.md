# 出题笔记：omni-qwen2audio-token-count

## 回源

HF transformers `processing_qwen2_audio.py`（feature_attention_mask 逐条传入、`<|AUDIO|>` 占位）+ `modeling_qwen2_audio.py` `Qwen2AudioEncoder._get_feat_extract_output_lengths`：

```python
input_lengths = (input_lengths - 1) // 2 + 1
output_lengths = (input_lengths - 2) // 2 + 1
```

展开逻辑在模型 forward（feature_attention_mask 求和 → conv 公式 → 替换占位）。

## 关键口径判断

- **先池化后 conv**：`sum(mask)` 再进公式，不能拿 `len(mask_row)` 进——padding 不变性单独成测。
- **audio_len=0 是合法状态**：mel ≤ 2 帧被两层 stride-2 conv 压没，占位替换为空。测试钉住 `expanded_len = len − 1` 的"净删"语义。
- **对应关系按音频条目序而非占位出现序**：官方 processor 把音频按条目填 feature，占位只是标记；`check_multi_audio_order` 钉住（乱序占位 + 有序 mask 行）。
- 边界手算曾错：L=5/6 我猜 2，实算 1（(5−1)//2+1=3 → (3−2)//2+1=1）。教训：conv 链每层都要实算，不能凭感觉。

## 边界划定

- 不做真实 mel/whisper encoder；不做 conv 权重与数值前向（那是框架算子题）。
- 与 674 `omni-tmrope-position` 分工：那题算"展开后位置 id 怎么排"，本题算"展开多少个"。两题公式同源（同 conv 结构），互为参照。
- 不含 video/audio-in-video 交错（qwen2_5_omni 范畴）。

## 测试设计

- oracle `_py` 独立重放（逐行 for 求和、两步 conv 拆开写）。
- 8 个 check：锚点、padding 不变性、conv 边界表（8 档）、audio_len=0 净删、多音频序、随机对拍 150 组、非法输入 6 档。
- 变异 9/9：conv 两层各 ±1/换 floor、不做 conv、占位计数 ±1、expanded 不减占位、不池化（用 len(row)）、pooled≥1 不查、mask 值不查、token id 放宽。`zero-allowed` 首跑存活（错误被计数校验提前拦），补"占位数恰好匹配但含 id=1"用例后全抓。
