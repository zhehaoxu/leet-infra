import torch


def solve(x, W0, b0, spectrum, n_frequency, random_loc_seed, scaling):
    """WaveFT 稀疏频谱增量（对齐 peft tuners/waveft.py 中 get_delta_weight 的 direct 路径）。
    x:(...,d_in)  W0:(d_out,d_in)  b0:(d_out,) 或 None。
    spectrum:(n_frequency,) 可学频谱系数；1 <= n_frequency <= d_out*d_in。
    坐标：generator = torch.Generator().manual_seed(random_loc_seed)，
          flat = torch.randperm(d_out*d_in, generator=generator)[:n_frequency]；
          delta[flat // d_in, flat % d_in] = spectrum；再整体乘 scaling。
    返回 (y, delta)：y = x @ W0.T + b0 + x @ delta.T，delta 与 W0 同形。
    内部 float64、返回 float32。禁止调用现成 PEFT 库。"""
    # TODO
    raise NotImplementedError
