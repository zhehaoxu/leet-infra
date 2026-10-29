def solve(scheme, samples, logit_mean=None, logit_std=None, mode_scale=None):
    """计算 SD3 训练的时间步采样密度：把一个批次的抽样映射成 [0, 1] 上的连续时间位置。

    scheme:三者之一——
        "logit_normal":samples 是标准正态抽样 z，输出 sigmoid(logit_mean + logit_std * z)；
        "mode":samples 是 [0, 1] 上的均匀抽样 u，输出 1 - u - mode_scale * (cos(pi*u/2)^2 - 1 + u)；
        "uniform":samples 原样返回。
    logit_mean / logit_std:logit_normal 必填，取有限实数且 logit_std >= 0；
    mode_scale:mode 必填，取有限非负实数。

    返回与 samples 同形状的一维 float32 张量（时间位置；乘以 num_train_timesteps 即得整数时间步）。
    非法输入（未知 scheme、samples 非一维张量、取值越界或非有限、缺少对应参数）抛 ValueError。
    """
    # TODO
    raise NotImplementedError
