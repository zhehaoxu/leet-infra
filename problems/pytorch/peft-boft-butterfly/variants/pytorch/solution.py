"""参考答案：BOFT 蝴蝶分解正交适配器。"""

import torch


def _float_tensor(name, value, ndim_min, ndim_max=None):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} 必须是 torch.Tensor")
    if value.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} 必须是 float32/float64 张量")
    if value.dim() < ndim_min:
        raise ValueError(f"{name} 维度不足")
    if ndim_max is not None and value.dim() > ndim_max:
        raise ValueError(f"{name} 维度过高")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} 含 NaN/Inf")
    return value


def _butterfly_perm(dim, block, stage):
    """第 stage 级的置换向量：新位置 k 取原位置 perm[k]。"""
    r = block // 2
    seg = block * (2 ** stage)
    groups = seg // r
    order = [g for g in range(groups) if g % 2 == 0] + [g for g in range(groups) if g % 2 == 1]
    perm = list(range(dim))
    for start in range(0, dim, seg):
        perm[start:start + seg] = [start + order[k] * r + t for k in range(groups) for t in range(r)]
    return perm


def solve(x, W0, b0, boft_R, boft_s):
    x = _float_tensor("x", x, 1)
    W0 = _float_tensor("W0", W0, 2, 2)
    boft_R = _float_tensor("boft_R", boft_R, 4, 4)
    boft_s = _float_tensor("boft_s", boft_s, 2, 2)

    out_dim, in_dim = W0.shape
    if in_dim == 0 or out_dim == 0:
        raise ValueError("W0 不能有空维度")
    if x.shape[-1] != in_dim:
        raise ValueError("x 的最后一维必须等于 d_in")

    stages, blocks, height, width = boft_R.shape
    if stages <= 0 or blocks <= 0:
        raise ValueError("boft_R 的级数与块数必须为正")
    if height != width:
        raise ValueError("boft_R 的每个块必须是方阵")
    if height <= 0 or height % 2 != 0:
        raise ValueError("block size 必须是正偶数")
    if in_dim % height != 0:
        raise ValueError("d_in 必须能被 block size 整除")
    if blocks != in_dim // height:
        raise ValueError("boft_R 的块数必须等于 d_in // block size")
    if blocks % (2 ** (stages - 1)) != 0:
        raise ValueError("block 数必须能被 2**(stages-1) 整除，否则蝴蝶分段不均匀")

    if boft_s.shape != (out_dim, 1):
        raise ValueError("boft_s 形状必须是 (d_out, 1)")
    if b0 is not None:
        b0 = _float_tensor("b0", b0, 1, 1)
        if b0.numel() != out_dim:
            raise ValueError("b0 长度必须等于 d_out")

    r64 = boft_R.to(torch.float64)
    skew = 0.5 * (r64 - r64.transpose(-1, -2))
    eye_h = torch.eye(height, dtype=torch.float64, device=r64.device)
    # Q = (I + S)^(-1) (I - S)：solve(..., left=False) 就是右解
    q = torch.linalg.solve(eye_h + skew, eye_h - skew, left=False)

    identity = torch.eye(in_dim, dtype=torch.float64, device=r64.device)
    rotation = identity.clone()
    for i in range(stages):
        diag = torch.zeros(in_dim, in_dim, dtype=torch.float64, device=r64.device)
        for j in range(blocks):
            diag[j * height:(j + 1) * height, j * height:(j + 1) * height] = q[i, j]
        perm = _butterfly_perm(in_dim, height, i)
        p = torch.zeros(in_dim, in_dim, dtype=torch.float64, device=r64.device)
        p[torch.arange(in_dim), torch.tensor(perm, device=r64.device)] = 1.0
        rotation = p @ diag @ p.transpose(-1, -2) @ rotation

    weight = W0.to(torch.float64)
    rotated = (rotation @ weight.T).T
    scaled = rotated * boft_s.to(torch.float64)
    y = x.to(torch.float64) @ scaled.T
    if b0 is not None:
        y = y + b0.to(torch.float64)
    return y.to(torch.float32), rotation.to(torch.float32)
