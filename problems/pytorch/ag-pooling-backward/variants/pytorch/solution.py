"""参考答案：按窗口枚举，max 走 argmax 路由、avg/adaptive 走面积分摊。

max 池化的反向是「一对一散射」：在前向窗口里找最大值（并列取行主序最小下标），
把整份 dY 交给它，因此 dx 的非零元素个数恰等于窗口个数，且不涉及任何除法。
avg 与 adaptive_avg 是「一对多分摊」，区别只在窗口边界的求法：固定核是等长的
[i·s−p, i·s−p+k)，adaptive 是 floor/ceil 夹出的变长区间，除数必须用**实际长度**，
否则重叠窗口的梯度不守恒。
"""

import math

import torch


def _pair(v):
    if isinstance(v, (tuple, list)):
        return int(v[0]), int(v[1])
    return int(v), int(v)


def _bounds(n, out, idx):
    start = int(math.floor(idx * n / out))
    end = int(math.ceil((idx + 1) * n / out))
    if end <= start:
        end = start + 1
    return start, min(end, n)


def solve(x, dY, mode='max', kernel=2, stride=None, padding=0):
    if x.dim() != 4 or dY.dim() != 4:
        raise ValueError("x 与 dY 必须是 (B, C, H, W) 四维张量")
    if tuple(x.shape[:2]) != tuple(dY.shape[:2]):
        raise ValueError("x 与 dY 的 batch/通道必须一致")
    if mode not in ('max', 'avg', 'adaptive_avg'):
        raise ValueError("mode 只支持 'max' / 'avg' / 'adaptive_avg'")

    b, c, h, w = x.shape
    oh, ow = dY.shape[2], dY.shape[3]
    xd = x.double()
    yd = dY.double()
    dx = torch.zeros_like(xd)
    zero = torch.zeros(b, c, dtype=torch.float64)

    if mode == 'adaptive_avg':
        if oh == 0 or ow == 0:
            raise ValueError("输出尺寸不得为 0")
        for i in range(oh):
            hs, he = _bounds(h, oh, i)
            for j in range(ow):
                ws, we = _bounds(w, ow, j)
                area = float((he - hs) * (we - ws))
                dx[:, :, hs:he, ws:we] += yd[:, :, i:i + 1, j:j + 1] / area
        return dx.to(x.dtype)

    kh, kw = _pair(kernel)
    sh, sw = _pair(kernel if stride is None else stride)
    ph, pw = _pair(padding)
    if kh <= 0 or kw <= 0 or sh <= 0 or sw <= 0:
        raise ValueError("kernel 与 stride 必须为正")
    if mode == 'avg' and (ph != 0 or pw != 0):
        raise ValueError("'avg' 模式不支持 padding（分摊除数口径不确定）")
    if oh != (h + 2 * ph - kh) // sh + 1 or ow != (w + 2 * pw - kw) // sw + 1:
        raise ValueError("dY 的空间尺寸与 x/kernel/stride/padding 不符")

    for i in range(oh):
        h0 = i * sh - ph
        for j in range(ow):
            w0 = j * sw - pw
            valid = [(h0 + a, w0 + t) for a in range(kh) for t in range(kw)
                     if 0 <= h0 + a < h and 0 <= w0 + t < w]
            if not valid:
                continue
            if mode == 'max':
                stack = torch.stack([xd[:, :, r, col] for r, col in valid], dim=-1)
                pick = stack.argmax(dim=-1)      # 并列时 argmax 给最小下标 = 行主序最前
                for pos, (r, col) in enumerate(valid):
                    dx[:, :, r, col] += torch.where(pick == pos, yd[:, :, i, j], zero)
            else:
                area = float(kh * kw)
                for r, col in valid:
                    dx[:, :, r, col] += yd[:, :, i, j] / area

    return dx.to(x.dtype)
