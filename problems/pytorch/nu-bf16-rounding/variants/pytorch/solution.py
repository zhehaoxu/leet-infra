"""参考答案：bf16 = f32 的高 16 位，所以三种舍入都能写成「加一个常数再截断」。

rne 的加数是 0x7FFF + 保留末位：低 16 位若 > 0x8000 必进位；恰好 == 0x8000（ties）时，
末位为 1 会让 0x8000 + 0x7FFF + 1 = 0x10000 溢出进位，末位为 0 则 0x8000 + 0x7FFF = 0xFFFF
不进位——一步位运算就把「就近」和「ties 取偶」同时表达了。rna 只加 0x8000，ties 必进位。
rz 是纯掩码，幅值只减不增，所以 3.4e38 截断后落在 bf16 最大有限值 3.3895e38 而不是 inf。

NaN 必须单拎出来：0x7F800001（一个 signaling NaN 的模式）加 0x8000 会变成 0x7F808001，
高 16 位是 0x7F80 —— NaN 被舍成了 +inf。所以 NaN 走「保留符号 + 强制 quiet 位」的旁路。
"""

import torch

KEEP = 0xFFFF0000
HALF = 0x8000
QUIET = 0x00400000
MODES = ('rne', 'rz', 'rna')


def solve(x, rounding='rne'):
    if not isinstance(x, torch.Tensor):
        raise ValueError("x 必须是 torch.Tensor")
    if x.dtype != torch.float32:
        raise ValueError("x 必须是 float32 张量")
    if not isinstance(rounding, str) or rounding not in MODES:
        raise ValueError(f"未知 rounding: {rounding}")

    xc = x.contiguous()
    u = xc.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    if rounding == 'rz':
        r = u & KEEP
    elif rounding == 'rna':
        r = (u + HALF) & KEEP
    else:
        r = (u + 0x7FFF + ((u >> 16) & 1)) & KEEP
    # NaN 旁路：保留符号位并强制 quiet，避免舍入把 NaN 变成 ±inf
    r = torch.where(torch.isnan(xc), (u & KEEP) | QUIET, r)
    return r.to(torch.int32).view(torch.float32).reshape(tuple(x.shape))
