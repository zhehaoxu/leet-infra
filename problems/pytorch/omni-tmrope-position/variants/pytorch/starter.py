"""TMRoPE：文本+音频段的 3D 位置 id 账（starter）。

口径钉死（HF transformers qwen2_5_omni thinker `get_rope_index`，
use_audio_in_video=False、无 image/video 分支）：

- 特殊 token id：TEXT=0、AUDIO_START=1、AUDIO_PAD=2、AUDIO_END=3。
- 音频块固定为 [AUDIO_START, AUDIO_PAD, AUDIO_END] 三个输入 token
  （官方序列里 pad 只占 1 个位置，模型 forward 时才展开）。
- 位置账（三行相同）：
  * 文本 run（含尾部）：`arange(text_len) + st_idx`，
    st_idx = 前一段 max + 1（首段为 0）。
  * 音频块：bos 1 个、audio_len 个、eos 1 个，各为
    `arange(n) + st_idx`；audio_len = ((L-1)//2 + 1 - 2)//2 + 1，
    L 为该音频的 mel 帧数（audio_seqlens 提供的未池化长度）。
- mrope_position_delta = 三行全局 max + 1 − 展开后长度（官方 input_ids 在
  调用前已把每个 AUDIO_PAD 展开成 audio_len 个；文本+音频账恰好恒为 0——
  非零 delta 只由图像/视频的 t/h/w 共享位置产生）。
- 校验：AUDIO_PAD 的 audio_seqlens 必须 >= 1（整型）；
  音频块数与 audio_seqlens 数量一致；token id 必须在 0..3。
"""

TEXT = 0
AUDIO_START = 1
AUDIO_PAD = 2
AUDIO_END = 3


def audio_expand_len(mel_len: int) -> int:
    """audio encoder 两层 conv 后的输出帧数（每帧一个位置）。"""
    return ((mel_len - 1) // 2 + 1 - 2) // 2 + 1


def solve(input_ids, audio_seqlens):
    """返回 {"position_ids": [row_t, row_h, row_w], "mrope_delta": int}。

    三行相同；position_ids 长度 = 展开后长度 =
    len(input_ids) - 3*块数 + sum(2 + audio_len)。delta 按展开后长度计。
    """
    raise NotImplementedError
