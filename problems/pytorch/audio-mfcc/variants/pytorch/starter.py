import torch


def solve(logmel, n_mfcc, lifter=0, cmvn=False):
    """把对数 Mel 谱压成 MFCC。

    logmel:(T, M) 对数 Mel 能量；n_mfcc:保留的低频倒谱系数个数（1 <= n_mfcc <= M）；
    lifter:倒谱提升系数 L，0 表示不提升；cmvn:是否做倒谱均值方差归一。

    1) 正交 DCT-II（沿 Mel 轴）：c[k] = s_k * sum_n x[n] * cos(pi*k*(n+0.5)/M)，
       其中 s_0 = sqrt(1/M)，s_k = sqrt(2/M)（k >= 1）；
    2) lifter（L > 0 时）：c[k] *= 1 + (L/2) * sin(pi*k/L)；
    3) cmvn=True 时按时间轴做 z-score：逐系数减均值、除以总体标准差（+1e-8 防零）。

    返回 (T, n_mfcc) float32。
    """
    # TODO
    raise NotImplementedError
