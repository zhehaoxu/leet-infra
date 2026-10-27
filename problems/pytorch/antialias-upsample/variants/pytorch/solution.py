import math
import struct

import torch


def _f32(v):
    """把 Python float 圆整到 float32 精度（模拟原生 CPU 的 float 计算）。"""
    return struct.unpack('<f', struct.pack('<f', v))[0]


def _cubic_conv1(x, a):
    """Keys cubic 卷积核 |x|<1 分支。"""
    return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0


def _cubic_conv2(x, a):
    """Keys cubic 卷积核 1<=|x|<2 分支。"""
    return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a


def _dim_taps(n_in, n_out, align_corners, cubic, antialias, f32):
    """单维坐标映射：返回 n_out 个 (tap 索引列表, 权重列表)。

    f32=True 时所有中间量按 float32 精度舍入，复刻原生 CPU
    antialias / separable 上采样路径；f32=False 全程 float64。
    """
    rnd = _f32 if f32 else (lambda v: v)
    interp_size = 4 if cubic else 2
    if n_out == n_in:
        # 恒等尺寸：逐位透传（与原生一致）
        return [([i], [1.0]) for i in range(n_in)]
    if f32:
        if align_corners:
            scale = _f32(_f32(n_in - 1) / _f32(n_out - 1)) if n_out > 1 else 0.0
        else:
            scale = _f32(_f32(n_in) / _f32(n_out))
    else:
        if align_corners:
            scale = (n_in - 1) / (n_out - 1) if n_out > 1 else 0.0
        else:
            scale = n_in / n_out
    # 非抗锯齿 bicubic 用 a=-0.75（OpenCV 口径）；抗锯齿统一 a=-0.5（PIL 口径）
    a = -0.5 if antialias else -0.75

    def filt(x):
        x = abs(x)
        if cubic:
            if x < 1.0:
                w = _cubic_conv1(x, a)
            elif x < 2.0:
                w = _cubic_conv2(x, a)
            else:
                w = 0.0
        else:
            w = 1.0 - x if x < 1.0 else 0.0
        return rnd(w)

    rows = []
    if antialias:
        if scale >= 1.0:
            support = rnd(rnd(interp_size * 0.5) * scale)
            invscale = rnd(1.0 / scale)
        else:
            support = interp_size * 0.5
            invscale = 1.0
        max_interp_size = int(math.ceil(support)) * 2 + 1
        for i in range(n_out):
            # 注意：原生 AA 路径即使 align_corners=True 也用 half-pixel 中心
            center = rnd(scale * rnd(i + 0.5))
            xmin = max(int(rnd(rnd(center - support) + 0.5)), 0)
            xsize = min(int(rnd(rnd(center + support) + 0.5)), n_in) - xmin
            xsize = max(0, min(xsize, max_interp_size))
            ws = []
            tot = 0.0
            for j in range(xsize):
                arg = rnd(rnd(rnd((j + xmin) - center) + 0.5) * invscale)
                w = filt(arg)
                ws.append(w)
                tot = rnd(tot + w)
            if tot != 0.0:
                ws = [rnd(w / tot) for w in ws]
            rows.append(([xmin + j for j in range(xsize)], ws))
    else:
        for i in range(n_out):
            if align_corners:
                src = rnd(scale * rnd(i))
            else:
                src = rnd(rnd(scale * rnd(i + 0.5)) - 0.5)
                if not cubic and src < 0.0:
                    src = 0.0  # 线性核 half-pixel 下 src<0 clamp 到 0
            idx = min(math.floor(src), n_in - 1)
            lam = rnd(min(max(rnd(rnd(src) - rnd(idx)) if f32 else src - idx, 0.0), 1.0))
            if not cubic:
                idx1 = idx + (1 if idx < n_in - 1 else 0)
                rows.append(([idx, idx1], [rnd(1.0 - lam), lam]))
            else:
                x1 = lam
                x2 = rnd(1.0 - lam)
                coeffs = (
                    _cubic_conv2(rnd(x1 + 1.0), a),
                    _cubic_conv1(x1, a),
                    _cubic_conv1(x2, a),
                    _cubic_conv2(rnd(x2 + 1.0), a),
                )
                taps = [min(max(idx + j - 1, 0), n_in - 1) for j in range(4)]
                rows.append((taps, [rnd(c) for c in coeffs]))
    return rows


def _apply(x, rows, dim):
    """把单维重采样作用在 x 的 dim 维：构造 (n_out, n_in) 稀疏权重矩阵后做矩阵乘。

    恒等尺寸时显式填单位阵，保证输出与输入逐位相等。
    累加在 float64 中完成（权重先已按输入 dtype 精度舍入）。
    """
    n_in = x.shape[dim]
    n_out = len(rows)
    M = torch.zeros(n_out, n_in, dtype=torch.float64)
    identity = all(len(t) == 1 and t[0] == k and w[0] == 1.0
                   for k, (t, w) in enumerate(rows))
    if identity:
        for k in range(n_in):
            M[k, k] = 1.0
    else:
        for k, (taps, ws) in enumerate(rows):
            for tp, w in zip(taps, ws):
                M[k, tp] += w  # 同索引权重累加（tap clamp 后可能出现）
    y = x.movedim(dim, -1) @ M.t().to(x.dtype)
    return y.movedim(-1, dim)


def solve(x, size, mode="bilinear", antialias=False, align_corners=False):
    """与 F.interpolate(mode=..., antialias=..., align_corners=...) 的 CPU 行为逐点对齐，
    语义契约见 starter.py 的 docstring。"""
    if mode not in ("bilinear", "bicubic", "trilinear"):
        raise ValueError(f"mode 仅支持 'bilinear' / 'bicubic' / 'trilinear'，got {mode!r}")
    if antialias and mode == "trilinear":
        raise ValueError("antialias=True 仅支持 bilinear / bicubic（4-D 输入）")
    if not isinstance(x, torch.Tensor) or x.dtype not in (torch.float32, torch.float64):
        raise ValueError("x 必须是 float32 / float64 张量")
    ndim = x.dim()
    need = 5 if mode == "trilinear" else 4
    if ndim != need:
        raise ValueError(f"mode={mode} 需要 {need}-D 输入，got {ndim}-D")
    if len(size) != need - 2:
        raise ValueError(f"size 长度应为 {need - 2}，got {len(size)}")
    size = [int(s) for s in size]
    if any(s <= 0 for s in size):
        raise ValueError(f"size 各维必须为正整数，got {size}")
    f32 = x.dtype == torch.float32
    cubic = mode == "bicubic"
    xd = x.to(torch.float64)
    for k, dim in enumerate(range(2, ndim)):
        rows = _dim_taps(xd.shape[dim], size[k], align_corners, cubic,
                         antialias and mode != "trilinear", f32)
        xd = _apply(xd, rows, dim)
    return xd.to(x.dtype)
