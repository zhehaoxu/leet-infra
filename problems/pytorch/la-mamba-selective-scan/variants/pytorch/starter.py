import torch
import torch.nn.functional as F


def solve(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False):
    """逐行转写 mamba_ssm 的 selective_scan_ref（实数 A、B/C 支持静态与变体两分支）。

    参数
    ----
    u     : [batch, dim, L]      输入
    delta : [batch, dim, L]      步长 Δ
    A     : [dim, dstate]        实数状态矩阵（复数不支持，抛 ValueError）
    B     : [dim, dstate]（静态）或 [batch, dstate, L]（变体）
    C     : 同 B 的两分支
    D     : None 或 [dim]        跳连
    z     : None 或 [batch, dim, L]  门控
    delta_bias : None 或 [dim]
    delta_softplus : bool，对 delta 先加 bias 再 softplus

    返回
    ----
    dict：
        out        [batch, dim, L]   与 u 同 dtype
        last_state [batch, dim, dstate]  最后一步的状态 x

    递推（全部 float() 化后计算）：
        deltaA   = exp(delta ⊗ A)                     # 'bdl,dn->bdln'
        deltaB_u = delta ⊗ B ⊗ u                      # 静态 'bdl,dn,bdl->bdln'
                                                      # 变体 'bdl,bnl,bdl->bdln'
        x = 0
        for i in range(L):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            y_i = x·C（静态 'bdn,dn->bd' / 变体 'bdn,bn->bd'，取 C[:, :, i]）
        out = y（堆叠 dim=2）；D 非 None 时 out += u * D[:, None]
        z 非 None 时 out = out * silu(z)
        last_state = 终态 x
    """
    raise NotImplementedError
