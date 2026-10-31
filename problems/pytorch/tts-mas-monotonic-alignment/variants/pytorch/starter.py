def solve(value):
    """Monotonic Alignment Search（Glow-TTS/FastSpeech2 口径）的确定性 max 对齐账。

    输入
    ----
    value : [T_mel, T_text] 对数似然矩阵（帧 t 对齐 token s 的 log P）

    记账（题面钉死，无跳搜的朴素单调对齐）：
      路径从 (0,0) 走到 (T_mel-1, T_text-1)：每帧恰好消耗一个 mel 帧（t 必 +1），
      text 指针不动（同 token 延续）或 +1（切下一 token）。即
        dp[t][s] = value[t][s] + max(dp[t-1][s], dp[t-1][s-1])
      dp[0][0] = value[0][0]，其余首行/首列不可达为 -inf；
      终点 (T_mel-1, T_text-1) 必须恰好是最后一个 token（每 token 至少 1 帧）。
      回溯得硬对齐 attn[t][s] ∈ {0,1}（帧 t 归属 token s），
      durations[s] = 归属 token s 的帧数。

    返回
    ----
    dict：
      log_prob   float          最优路径累计 log 似然（dp 终点值）
      attn       [[0/1]]        [T_mel, T_text] 硬对齐矩阵
      durations  list[int]      每 token 帧数（和恒为 T_mel）
    """
    raise NotImplementedError
