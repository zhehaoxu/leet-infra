import math

import numpy as np
import torch


def _bilinear_np(feat_c, y, x):
    H, W = feat_c.shape
    y0 = math.floor(y); x0 = math.floor(x)
    y1 = y0 + 1; x1 = x0 + 1
    ly = y - y0; lx = x - x0

    def val(a, b):
        if 0 <= a < H and 0 <= b < W:
            return feat_c[a, b]
        return 0.0

    return ((1 - ly) * (1 - lx) * val(y0, x0) + (1 - ly) * lx * val(y0, x1) +
            ly * (1 - lx) * val(y1, x0) + ly * lx * val(y1, x1))


def _roi_align_numpy(features, rois, pooled_size, spatial_scale, sampling_ratio, aligned):
    """独立 numpy 参考实现（与题目规则一致），用于核对 solve()。"""
    N, C, H, W = features.shape
    K = rois.shape[0]
    out = np.zeros((K, C, pooled_size, pooled_size), dtype=np.float64)
    r = max(1, int(sampling_ratio))
    for k in range(K):
        b = int(rois[k][0])
        x1 = rois[k][1] * spatial_scale; y1 = rois[k][2] * spatial_scale
        x2 = rois[k][3] * spatial_scale; y2 = rois[k][4] * spatial_scale
        if aligned:
            x1 -= 0.5; y1 -= 0.5; x2 -= 0.5; y2 -= 0.5
        bin_w = (x2 - x1) / pooled_size; bin_h = (y2 - y1) / pooled_size
        fmap = features[b]
        for i in range(pooled_size):
            ys = y1 + i * bin_h
            for j in range(pooled_size):
                xs = x1 + j * bin_w
                acc = np.zeros(C, dtype=np.float64)
                for sy in range(r):
                    fy = ys + (sy + 0.5) * bin_h / r
                    for sx in range(r):
                        fx = xs + (sx + 0.5) * bin_w / r
                        for c in range(C):
                            acc[c] += _bilinear_np(fmap[c], fy, fx)
                out[k, :, i, j] = acc / (r * r)
    return out


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_edge_example(solve):
    ex = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4)
    rois = torch.tensor([[0.0, 0.0, 0.0, 3.0, 3.0]])
    got = solve(ex, rois, 2, 1.0, 2, False).numpy().astype(np.float64)
    ref = np.array([[[[3.75, 5.25], [9.75, 11.25]]]])
    assert got.shape == (1, 1, 2, 2)
    assert np.allclose(got, ref, atol=1e-5), got


def check_matches_numpy_random(solve):
    torch.manual_seed(0)
    feats = torch.randn(2, 3, 7, 7)
    rois = torch.tensor([
        [0.0, 0.0, 0.0, 28.0, 28.0],
        [1.0, 3.0, 2.0, 20.0, 26.0],
        [0.0, 8.0, 8.0, 16.0, 16.0],
    ])
    fn = feats.numpy().astype(np.float64)
    rn = rois.numpy().astype(np.float64)
    for P, s, r, al in [(2, 0.25, 2, True), (3, 1.0, 1, False), (2, 0.5, 4, True)]:
        got = solve(feats, rois, P, s, r, al).numpy().astype(np.float64)
        ref = _roi_align_numpy(fn, rn, P, s, r, al)
        assert np.allclose(got, ref, atol=1e-5), f"P={P} s={s} r={r} aligned={al}"


def check_aligned_differs(solve):
    # aligned=True 与 False 因 0.5 偏移应给出不同结果
    torch.manual_seed(1)
    feats = torch.randn(1, 2, 6, 6)
    rois = torch.tensor([[0.0, 2.0, 2.0, 14.0, 14.0]])
    a0 = solve(feats, rois, 2, 0.25, 2, False).numpy()
    a1 = solve(feats, rois, 2, 0.25, 2, True).numpy()
    assert not np.allclose(a0, a1, atol=1e-4)
    assert np.allclose(a1, _roi_align_numpy(feats.numpy().astype(np.float64),
                                            rois.numpy().astype(np.float64), 2, 0.25, 2, True), atol=1e-5)


def check_single_bin_is_mean(solve):
    # pooled_size=1, sampling_ratio=1：单个采样点在 bin 中心的双线性值
    torch.manual_seed(2)
    feats = torch.randn(1, 2, 5, 5)
    rois = torch.tensor([[0.0, 0.0, 0.0, 20.0, 20.0]])
    got = solve(feats, rois, 1, 0.25, 1, False).numpy().astype(np.float64)
    ref = _roi_align_numpy(feats.numpy().astype(np.float64), rois.numpy().astype(np.float64), 1, 0.25, 1, False)
    assert np.allclose(got, ref, atol=1e-5)


check_edge_example.example = True
