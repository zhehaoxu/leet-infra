def solve(eps_cond, eps_uncond, guidance_scale, step, zero_init_steps=1, eps=1e-8):
    """CFG-Zero*：前若干步零初始化 + 最小二乘尺度校正的引导。

    eps_cond / eps_uncond:同形状预测，第 0 维是 batch；
    guidance_scale:s，外推基准是校正后的 eps_uncond；
    step:当前去噪步索引（从 0 起，整数且非负）；
    zero_init_steps:前 k 步直接返回全零预测（k=0 表示不零初始化）；
    eps:最小二乘尺度分母的稳定项（正有限实数）。

    返回 float32 张量，形状与 eps_cond 相同。
    """
    # TODO
    raise NotImplementedError
