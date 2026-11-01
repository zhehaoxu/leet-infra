def solve(emission, tokens, transcript, separator="|"):
    """WhisperX 强制对齐账：trellis DP + 回溯 + merge_repeats/merge_words。

    emission: (T, V) log 概率张量；tokens: 词表下标列表；
    transcript: 等长标签列表；separator: 词分隔标签（默认 "|"）。

    官方口径（whisperx/alignment.py torch fallback 原样）：
      trellis (T+1, N+1)：[0,0]=0；列 0 = blank 发射的 cumsum；
      第 0 行词表列 = -inf；**列 0 尾部 N 行 = +inf**（强制收尾）；
      递推 max(stay, change)，log 域。
      回溯：t_start = argmax(trellis[:, N])（并列取首行）；
      lbl 按 changed > stayed（**严格 >**，平分留 blank）；
      帧概率 = exp(emission[t-1, lbl])（概率域，与 trellis 的 log 域
      混用是官方原样）；j 未归 0 走完循环 -> None（T < N 时必现）。
      merge_repeats：连续同 token 段 -> [首帧, 末帧+1)，score = 帧均。
      merge_words：按 separator 切分，词分 = 长度加权（非帧均）；
      首/连续/尾 separator 不产空词。

    返回 dict：ok / path / trellis_corner / col0_inf / segments / words
    红线：不得调用 torchaudio / torio / numba / whisperx。
    """
    raise NotImplementedError
