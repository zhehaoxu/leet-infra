import zlib


def solve(windows, temperatures=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
          compression_ratio_threshold=2.4, logprob_threshold=-1.0,
          no_speech_threshold=0.6):
    """Whisper transcribe 的窗口级质量门控与温度回退账（official openai/whisper 口径）。

    输入
    ----
    windows : 非空 list[dict]，每窗记录各温度的解码指标：
              {'texts': [str] * T_n,        # 每温度的解码文本（算压缩比）
               'avg_logprob': [float] * T_n,
               'no_speech_prob': [float] * T_n}
              T_n = len(temperatures)
    temperatures : 温度回退序列（非空 list[float]）
    compression_ratio_threshold / logprob_threshold / no_speech_threshold :
              官方默认 2.4 / -1.0 / 0.6

    记账（题面钉死，与官方 decode_with_fallback + 段级 skip 一致）：
      1. compression_ratio(text) = len(utf8 字节) / len(zlib.compress(utf8 字节))；
      2. 逐温度回退（decode_with_fallback）：
         needs_fallback = cr > cr_t 或 avg_logprob < lp_t；
         若 no_speech_prob > ns_t 且 avg_logprob < lp_t → needs_fallback = False
         （silence 覆盖，优先级最高）；
         not needs_fallback → 选中当前温度并停止；全部温度用尽 → 选中最后一个；
      3. 段级 skip（用选中温度的指标）：
         should_skip = no_speech_prob > ns_t；
         若 avg_logprob > lp_t（严格大于）→ 不 skip（豁免）；
         skip 的窗口 action='skip'，否则 'transcribe'。

    返回
    ----
    list[dict]，每窗：
      compression_ratios    list[float]  每温度的 zlib 压缩比
      selected_temp_index   int          选中的温度下标
      fallback_chain        list[int]    实际尝试的温度下标序列
      silence_covered       bool         选中温度上 needs_fallback 被 silence 覆盖
      action                str          'transcribe' | 'skip'
    """
    raise NotImplementedError
