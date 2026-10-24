import math

import numpy as np
import torch


def _roi_pool_numpy(features, rois, pooled_size, spatial_scale):
    """独立 numpy 参考实现（与题目规则一致），用于核对 solve()。"""
    N, C, H, W = features.shape
    K = rois.shape[0]
    out = np.zeros((K, C, pooled_size, pooled_size), dtype=np.float64)
    for k in range(K):
        b = int(rois[k][0]); x1, y1, x2, y2 = rois[k][1], rois[k][2], rois[k][3], rois[k][4]
        fx1 = int(math.floor(x1 * spatial_scale)); fy1 = int(math.floor(y1 * spatial_scale))
        fx2 = int(math.floor(x2 * spatial_scale)); fy2 = int(math.floor(y2 * spatial_scale))
        for i in range(pooled_size):
            ylo = fy1 + (fy2 - fy1) * i // pooled_size
            yhi = fy1 + (fy2 - fy1) * (i + 1) // pooled_size
            for j in range(pooled_size):
                xlo = fx1 + (fx2 - fx1) * j // pooled_size
                xhi = fx1 + (fx2 - fx1) * (j + 1) // pooled_size
                if yhi <= ylo or xhi <= xlo:
                    continue
                yl, yh = max(0, ylo), min(H, yhi)
                xl, xh = max(0, xlo), min(W, xhi)
                if yh <= yl or xh <= xl:
                    continue
                for c in range(C):
                    out[k, c, i, j] = features[b, c, yl:yh, xl:xh].max()
    return out


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_edge_example(solve):
    ex = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4)
    rois = torch.tensor([[0.0, 0.0, 0.0, 3.0, 3.0]])
    got = solve(ex, rois, 2, 1.0)
    ref = np.array([[[[0.0, 2.0], [8.0, 10.0]]]])
    assert got.shape == (1, 1, 2, 2)
    assert np.allclose(got.numpy(), ref, atol=1e-5), got.numpy()


def check_matches_numpy_random(solve):
    torch.manual_seed(0)
    feats = torch.randn(2, 3, 7, 7)
    rois = torch.tensor([
        [0.0, 0.0, 0.0, 28.0, 28.0],
        [1.0, 3.5, 2.0, 20.0, 26.5],
        [0.0, 8.0, 8.0, 16.0, 16.0],
    ])
    for P, s in [(2, 0.25), (3, 1.0), (1, 0.5)]:
        got = solve(feats, rois, P, s).numpy().astype(np.float64)
        ref = _roi_pool_numpy(feats.numpy().astype(np.float64), rois.numpy().astype(np.float64), P, s)
        assert np.allclose(got, ref, atol=1e-5), f"P={P} s={s}"


def check_out_of_bounds_clamped(solve):
    # ROI 越界应被裁剪到特征图边界，而非报错
    torch.manual_seed(1)
    feats = torch.randn(1, 2, 5, 5)
    rois = torch.tensor([[0.0, -4.0, -4.0, 40.0, 40.0]])
    got = solve(feats, rois, 2, 1.0)
    ref = _roi_pool_numpy(feats.numpy().astype(np.float64), rois.numpy().astype(np.float64), 2, 1.0)
    assert np.allclose(got.numpy(), ref, atol=1e-5)


def check_multi_batch_selection(solve):
    # 两个 ROI 指向不同 batch 图，输出应各取对应图特征
    torch.manual_seed(2)
    feats = torch.randn(2, 2, 6, 6)
    rois = torch.tensor([[0.0, 0.0, 0.0, 24.0, 24.0],
                         [1.0, 0.0, 0.0, 24.0, 24.0]])
    got = solve(feats, rois, 2, 0.25).numpy().astype(np.float64)
    ref = _roi_pool_numpy(feats.numpy().astype(np.float64), rois.numpy().astype(np.float64), 2, 0.25)
    assert np.allclose(got, ref, atol=1e-5)
    assert not np.allclose(got[0], got[1]), "不同 batch 应得到不同特征"


check_edge_example.example = True
