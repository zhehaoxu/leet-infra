def solve(latents, scaling_factor=1.0, shift_factor=0.0, channel_mean=None, channel_std=None, inverse=False):
    """VAE latent 与扩散模型 latent 之间的换算（SDXL / Flux 的 scale·(x - shift) 与先验白化）。

    第 0 维是 batch，第 1 维是通道；channel_mean / channel_std 为每通道一个值（或标量）。
    inverse=False（默认，解码方向）：先做先验校正 raw = latents·std + mean，再 x = raw / scaling + shift；
    inverse=True（编码方向）：先 raw = (x - shift)·scaling，再 latents = (raw - mean) / std。

    返回 float32 张量，形状与输入相同。channel_mean 与 channel_std 必须同时给或同时不给。
    """
    # TODO
    raise NotImplementedError
