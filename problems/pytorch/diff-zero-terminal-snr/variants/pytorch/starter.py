def solve(betas):
    """把一条 β 序列重标定成「终端 SNR 为零」，并给出 log-SNR 与 v 预测的旋转角。

    betas:一维 float 张量，元素落在 [0, 1)，索引 0 = 最干净、末位 = 最噪声；
    1) 原始累积乘积 \barα_t = Π(1 - β_i)；
    2) 零终端 SNR 重标定：\barα'_t = (\barα_t - \barα_T) / (1 - \barα_T)，使 \barα'_{T} = 0；
    3) λ_t = log(\barα'_t / (1 - \barα'_t))，两端分别取 -inf / +inf；
    4) θ_t = atan2(√(1-\barα'_t), √\barα'_t)，即 ε → v 的旋转角（v = cosθ·ε - sinθ·x₀）。

    返回 (alpha_bars, log_snrs, angles)，三者均为同长度 float32 张量。
    """
    # TODO
    raise NotImplementedError
