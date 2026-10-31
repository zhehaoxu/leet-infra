"""Qwen2-Audio：音频占位 token 展开数账（starter）。

口径钉死（HF transformers qwen2_audio processing/modeling）：

- `feature_attention_mask` 形状 [num_audios, mel_max]，0/1；每行求和
  得到该音频**未池化** mel 帧数（右侧 padding 全部消去）。
- 展开长度 = 两层 conv 输出：`audio_len = ((l−1)//2 + 1 − 2)//2 + 1`
  （l 为池化后帧数）；l ≤ 2 时 audio_len = 0。
- `input_ids` 中音频占位 token（id=AUDIO_TOKEN=2）出现次数必须等于
  音频数（processor 的 feature 口径：每个 `<|AUDIO|>` 对应一条音频）。
- 输出：
  * pooled_lens：每条音频的池化帧数；
  * audio_lens：每条音频的展开 token 数；
  * expanded_len：展开后序列长度 = len(input_ids) − 占位数 + Σaudio_len
    （占位 token 被替换，多余位置删除）。
- 校验：mask 值必须 0/1；pooled ≥ 1；占位数 == 音频数；input_ids 只含
  0（文本）/2（音频占位）。
"""

AUDIO_TOKEN = 2


def conv_out_len(mel_len: int) -> int:
    """两层 conv（stride 2、去首尾帧）后的输出帧数。"""
    return ((mel_len - 1) // 2 + 1 - 2) // 2 + 1


def solve(input_ids, feature_attention_mask):
    """返回 {"pooled_lens": [...], "audio_lens": [...], "expanded_len": int}。"""
    raise NotImplementedError
