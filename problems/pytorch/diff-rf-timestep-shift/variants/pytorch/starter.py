def solve(
    times,
    shift=None,
    mu=None,
    seq_len=None,
    base_shift=0.5,
    max_shift=1.15,
    base_seq_len=256,
    max_seq_len=4096,
):
    """把 [0, 1] 上的采样时间栅格按 shift 映射到噪声侧（SD3 / Flux 的 timestep shift）。

    times:一维 float 张量，取值 [0, 1]，1 = 纯噪声、0 = 干净数据；
    shift / mu / seq_len:三者恰给一个——直接给 s，或给 µ（s = exp(µ)），
        或给图像序列长度由「插值出 µ」推出 s：µ = m·seq_len + b，
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)，
        b = base_shift - m·base_seq_len；
    映射：t' = s·t / (1 + (s - 1)·t)（等价于 logit 域平移 µ：t' = σ(logit(t) + µ)）。

    返回 (shifted, s)：shifted 与 times 同形状的 float32 张量，s 为 python float。
    """
    # TODO
    raise NotImplementedError
