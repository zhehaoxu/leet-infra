import torch


def solve(x, up, down, taps=63, beta=8.0):
    """整数比 L/M 多相重采样（有理数倍率，windowed-sinc 原型滤波器）。

    x:(T,) 单声道波形；up/down:L 与 M（目标采样率 = 源采样率 * up / down）；
    taps:原型滤波器长度（奇数）；beta:Kaiser 窗的 beta 参数。

    原型低通（按「上采样后」的栅格设计，中心偏移 c = (taps-1)/2）：
        fc = 0.5 / max(up, down)，对**有符号偏移** j = -c..c
        h(j) = 2*fc*sinc(2*fc*j) * kaiser(taps, beta)(j+c)，
        再整体缩放使 sum(h) = up（保直流增益为 1）。
    多相输出（边界外一律按 0 处理，不补边）：
        y[m] = sum_n x[n] * h(m*down - n*up)，只累加 |m*down - n*up| <= c 的项；
        输出长度 out_len = ((T-1)*up) // down + 1。

    返回 y:(out_len,) float32，float64 计算、最后转回。
    """
    # TODO
    raise NotImplementedError
