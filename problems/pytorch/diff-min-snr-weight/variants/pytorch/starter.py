def solve(alphas_cumprod, timesteps, gamma):
    """按 Min-SNR 策略计算每个时间步的 SNR 与损失权重。

    alphas_cumprod:一维浮点张量，长度 T >= 1，每个元素严格落在 (0, 1) 内（噪声调度的累积 alpha）；
    timesteps:一维整型张量（int32 / int64），每个元素是 [0, T) 的下标；
    gamma:正的有限实数，SNR 的上限（常用 5.0）。

    先算 snr = (sqrt(alpha_bar) / sqrt(1 - alpha_bar))^2，再算 weight = min(snr, gamma) / snr。
    返回 (snr, weight)：与 timesteps 同形状的 float32 张量。
    非法输入（维度/类型不对、alphas_cumprod 越界或非有限、timesteps 越界、gamma 非正）抛 ValueError。
    """
    # TODO
    raise NotImplementedError
