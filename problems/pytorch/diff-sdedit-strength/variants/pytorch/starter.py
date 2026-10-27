def solve(x0, noise, alpha_bars, strength):
    """SDEdit 的「加噪到强度 s 对应的时间步，再从那里往回采样」这一步的起点计算。

    x0:干净样本；noise:同一形状的 ε（显式入参，不现采）；
    alpha_bars:一维 float 张量，按噪声级升序排列（索引 0 最干净、末位最噪声），
        要求严格递减且落在 (0, 1]；
    strength:s∈[0, 1]，起始索引取 i = floor(s·(T-1))，T = len(alpha_bars)。

    x_noisy = √ᾱ_i·x0 + √(1 - ᾱ_i)·noise，返回 (x_noisy, i)，前者 float32、后者 int。
    """
    # TODO
    raise NotImplementedError
