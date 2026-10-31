import torch


def solve(keys, group_size=128, rbits=4, tbits=4):
    """PolarQuant：把 KV 的 head_dim 按「前后两半」配成极坐标 (角度, 半径)，分组量化后打包。

    keys       : float32/float64 的 (B, H, L, D)，D 必须是偶数、L 必须是 group_size 的正整数倍
    group_size : 正整数，沿 token 轴切连续组，每组 G 个 token
    rbits      : 正整数，半径的码字位宽（级数 2**rbits）
    tbits      : 正整数，角度的码字位宽（级数 2**tbits）

    口径（内部一律 float64，返回张量按 keys 的 dtype；只读入参，不就地修改）：
      Dh = D // 2，第 d 个「对」是 (keys[..., d], keys[..., Dh + d])（两半配对，不是奇偶交错）。
      phi = atan2(后半, 前半)，负值 +2π 折到 [0, 2π)；radii = 该对模长。
      每个 (B, H, 组, d) 只在该组的 G 个 token 上取 min/max：
        scale = (max - min) / 2**bits —— 分母是 2**bits，不是 2**bits - 1；
        code = clamp(floor((x - min) / scale), 0, 2**bits - 1) —— 向下取整，不是四舍五入；
        scale == 0（组内常数）时码字取 0。
      indices = (rho << tbits) + theta。
      重构取 bin 中心：hat = (code + 0.5) * scale + min，
        recon[..., d] = radii_hat * cos(phi_hat)，recon[..., Dh + d] = radii_hat * sin(phi_hat)。

    返回 dict，键固定为 6 个：
      phi     : (B, H, L//G, G, Dh) 折回 [0, 2π) 的角度
      radii   : (B, H, L//G, G, Dh) 成对模长
      theta   : (B, H, L//G, G, Dh) int64 角度码字
      rho     : (B, H, L//G, G, Dh) int64 半径码字
      indices : (B, H, L//G, G, Dh) int64 打包后的码
      recon   : (B, H, L, D)        反量化重建

    keys 非四维 / 非 float32、float64 / D 为奇数 / L 不是 group_size 的正整数倍 /
    group_size、rbits、tbits 不是正整数（bool 与 4.0 都算非法）时抛 ValueError。
    """
    # TODO
    raise NotImplementedError
