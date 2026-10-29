import torch


def solve(dY, x, weight, stride=1, padding=0):
    """单组卷积 Y = conv2d(x, weight) 的反向：返回 (dx, dw)。

    前向：Y[b,co,u,v] = Σ_ci Σ_k Σ_l x[b,ci,u·stride+k−padding,v·stride+l−padding] · weight[co,ci,k,l]。

    入参：
    - dY: (B, Cout, Hout, Wout)；x: (B, Cin, Hin, Win)；weight: (Cout, Cin, kh, kw)。
    - stride: int ≥ 1；padding: int ≥ 0；两者对 dx 与 dw 同时生效。
    - 只支持 groups=1；Cin 与 weight 的第 1 轴不符、dY 的空间尺寸与前向公式不符
      （Hout != (Hin+2p−kh)//stride+1）都抛 ValueError。

    口径：
    - dx 是 dY 与**旋转 180° 的核**的相关（等价于全卷积），越出 x 边界的贡献直接丢弃；
      padding 位置的梯度**不**写进 dx（dx 形状始终等于 x 形状）。
    - dw[co,ci,k,l] = Σ_b Σ_{u,v} dY[b,co,u,v] · x[b,ci,u·stride+k−padding,v·stride+l−padding]，
      只对落在 x 范围内的 (u,v) 求和，形状等于 weight 形状，batch 轴一定要折叠掉。
    - 内部 float64 累计，返回 dtype 分别与 x、weight 一致。
    """
    # TODO
    raise NotImplementedError
